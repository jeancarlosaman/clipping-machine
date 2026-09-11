"""Exercises app.workers.upload.run() directly -- real DB, real object
storage for the clip's video bytes, but app.core.tiktok_client's actual
HTTP calls are monkeypatched (see test_tiktok_client.py for coverage of
that module's own HTTP/error handling). Same spirit as
test_rendering_worker.py mocking ffmpeg's subprocess call via a spy while
keeping everything else real.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core import tiktok_client
from app.core.config import settings
from app.core.crypto import decrypt_token, encrypt_token
from app.core.storage import storage
from app.core.tiktok_client import TikTokAPIError
from app.db.models import CandidateSegment, CreatorAccount, RenderedClip, StreamJob, UploadAuditLog, UploadTask
from app.workers import upload


def _seed_task(db_session, user, tmp_path, *, token_expires_at=None, refresh_token=None, confidence_score=None):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake mp4 bytes for upload test")
    object_key = f"clips/{uuid.uuid4()}/clip.mp4"
    storage.put_file(str(video_path), object_key)

    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="ready_for_review")
    db_session.add(job)
    db_session.flush()

    segment = CandidateSegment(stream_job_id=job.id, start_seconds=0, end_seconds=10, status="selected")
    db_session.add(segment)
    db_session.flush()

    clip = RenderedClip(candidate_segment_id=segment.id, stream_job_id=job.id, status="rendered", object_key=object_key)
    db_session.add(clip)

    account = CreatorAccount(
        user_id=user.id,
        platform="tiktok",
        external_account_id="open-id-1",
        access_token_encrypted=encrypt_token("access-token-1"),
        refresh_token_encrypted=encrypt_token(refresh_token) if refresh_token else None,
        token_expires_at=token_expires_at,
    )
    db_session.add(account)
    db_session.flush()

    task = UploadTask(
        rendered_clip_id=clip.id, creator_account_id=account.id, platform="tiktok",
        target_mode="draft", status="queued", confidence_score=confidence_score,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    db_session.refresh(account)
    return task, account


def _patch_success(monkeypatch, *, publicaly_available_post_id=None):
    calls = {"init": 0, "chunks": [], "status": 0}

    def fake_init(access_token, *, video_size, chunk_size, total_chunk_count):
        calls["init"] += 1
        return {"publish_id": "pub_success_1", "upload_url": "https://upload.example/x"}

    def fake_chunk(upload_url, chunk_bytes, *, first_byte, last_byte, total_bytes, content_type="video/mp4"):
        calls["chunks"].append((first_byte, last_byte, total_bytes))

    def fake_status(access_token, publish_id):
        calls["status"] += 1
        data = {"status": "PUBLISH_COMPLETE"}
        if publicaly_available_post_id is not None:
            data["publicaly_available_post_id"] = [publicaly_available_post_id]
        return data

    monkeypatch.setattr(tiktok_client, "init_inbox_upload", fake_init)
    monkeypatch.setattr(tiktok_client, "upload_video_chunk", fake_chunk)
    monkeypatch.setattr(tiktok_client, "fetch_publish_status", fake_status)
    return calls


def test_upload_success_marks_uploaded_and_stores_post_id(db_session, user, tmp_path, monkeypatch):
    task, _account = _seed_task(db_session, user, tmp_path)
    calls = _patch_success(monkeypatch, publicaly_available_post_id=999888777)

    upload.run(str(task.id))

    db_session.refresh(task)
    assert task.status == "uploaded"
    assert task.publish_id == "pub_success_1"
    assert task.external_post_id == "999888777"
    assert task.last_error is None
    assert calls["init"] == 1
    assert len(calls["chunks"]) == 1  # small fake video -> single chunk
    events = [row.event for row in db_session.query(UploadAuditLog).filter_by(upload_task_id=task.id)]
    assert "attempt" in events
    assert "success" in events


def test_upload_resumes_existing_publish_id_without_reuploading(db_session, user, tmp_path, monkeypatch):
    task, _account = _seed_task(db_session, user, tmp_path)
    task.publish_id = "pub_already_started"
    db_session.commit()
    calls = _patch_success(monkeypatch)

    upload.run(str(task.id))

    db_session.refresh(task)
    assert task.status == "uploaded"
    assert task.publish_id == "pub_already_started"
    assert calls["init"] == 0  # never re-initiated
    assert calls["chunks"] == []  # never re-uploaded bytes
    assert calls["status"] == 1


def test_upload_marks_failed_on_terminal_tiktok_failure(db_session, user, tmp_path, monkeypatch):
    task, _account = _seed_task(db_session, user, tmp_path)
    monkeypatch.setattr(
        tiktok_client, "init_inbox_upload",
        lambda *a, **k: {"publish_id": "pub_fail_1", "upload_url": "https://upload.example/x"},
    )
    monkeypatch.setattr(tiktok_client, "upload_video_chunk", lambda *a, **k: None)
    monkeypatch.setattr(
        tiktok_client, "fetch_publish_status",
        lambda *a, **k: {"status": "FAILED", "fail_reason": "file_format_check_failed"},
    )

    upload.run(str(task.id))

    db_session.refresh(task)
    assert task.status == "failed"
    assert "file_format_check_failed" in task.last_error


def test_upload_non_retryable_error_marks_failed_without_raising(db_session, user, tmp_path, monkeypatch):
    task, _account = _seed_task(db_session, user, tmp_path)

    def raise_non_retryable(*a, **k):
        raise TikTokAPIError("access token invalid", retryable=False)

    monkeypatch.setattr(tiktok_client, "init_inbox_upload", raise_non_retryable)

    upload.run(str(task.id))  # must not raise -- RQ should not retry this

    db_session.refresh(task)
    assert task.status == "failed"
    assert "access token invalid" in task.last_error
    assert task.publish_id is None  # init never succeeded, nothing to resume


def test_upload_retryable_error_raises_and_leaves_status_uploading(db_session, user, tmp_path, monkeypatch):
    task, _account = _seed_task(db_session, user, tmp_path)

    def raise_retryable(*a, **k):
        raise TikTokAPIError("temporary TikTok 503", retryable=True)

    monkeypatch.setattr(tiktok_client, "init_inbox_upload", raise_retryable)

    with pytest.raises(TikTokAPIError):
        upload.run(str(task.id))  # must raise -- this is how RQ's Retry knows to re-enqueue

    db_session.refresh(task)
    assert task.status == "uploading"  # not marked failed -- on_failure only fires once RQ's retries are exhausted


def test_upload_still_processing_after_poll_budget_raises_retryable_and_keeps_publish_id(
    db_session, user, tmp_path, monkeypatch
):
    task, _account = _seed_task(db_session, user, tmp_path)
    monkeypatch.setattr(settings, "tiktok_status_poll_max_attempts", 2)
    monkeypatch.setattr(settings, "tiktok_status_poll_interval_seconds", 0)
    monkeypatch.setattr(
        tiktok_client, "init_inbox_upload",
        lambda *a, **k: {"publish_id": "pub_still_processing", "upload_url": "https://upload.example/x"},
    )
    monkeypatch.setattr(tiktok_client, "upload_video_chunk", lambda *a, **k: None)
    monkeypatch.setattr(tiktok_client, "fetch_publish_status", lambda *a, **k: {"status": "PROCESSING_UPLOAD"})

    with pytest.raises(TikTokAPIError) as exc_info:
        upload.run(str(task.id))
    assert exc_info.value.retryable is True

    db_session.refresh(task)
    # publish_id persisted even though this attempt didn't finish -- a
    # retry resumes polling it instead of re-uploading (see the
    # resume-without-reupload test above for that behavior itself).
    assert task.publish_id == "pub_still_processing"


def test_upload_refreshes_expired_token_before_calling_api(db_session, user, tmp_path, monkeypatch):
    task, account = _seed_task(
        db_session, user, tmp_path,
        token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        refresh_token="refresh-token-1",
    )
    seen_tokens = []

    def fake_refresh(refresh_token):
        assert refresh_token == "refresh-token-1"
        return {"access_token": "refreshed-access-token", "refresh_token": "refreshed-refresh-token", "expires_in": 86400}

    def fake_init(access_token, **kwargs):
        seen_tokens.append(access_token)
        return {"publish_id": "pub_refreshed", "upload_url": "https://upload.example/x"}

    monkeypatch.setattr(tiktok_client, "refresh_access_token", fake_refresh)
    monkeypatch.setattr(tiktok_client, "init_inbox_upload", fake_init)
    monkeypatch.setattr(tiktok_client, "upload_video_chunk", lambda *a, **k: None)
    monkeypatch.setattr(tiktok_client, "fetch_publish_status", lambda *a, **k: {"status": "SEND_TO_USER_INBOX"})

    upload.run(str(task.id))

    assert seen_tokens == ["refreshed-access-token"]
    db_session.refresh(account)
    assert decrypt_token(account.access_token_encrypted) == "refreshed-access-token"
    assert decrypt_token(account.refresh_token_encrypted) == "refreshed-refresh-token"


def test_upload_blocked_by_daily_cap_never_calls_tiktok(db_session, user, tmp_path, monkeypatch):
    task, account = _seed_task(db_session, user, tmp_path)
    account.daily_upload_cap = 1
    db_session.commit()
    # A prior "uploaded" task against the same account within 24h.
    other = UploadTask(
        rendered_clip_id=task.rendered_clip_id, creator_account_id=account.id,
        platform="tiktok", target_mode="draft", status="uploaded",
    )
    db_session.add(other)
    db_session.commit()

    called = []
    monkeypatch.setattr(tiktok_client, "init_inbox_upload", lambda *a, **k: called.append("init"))

    upload.run(str(task.id))

    db_session.refresh(task)
    assert task.status == "blocked"
    assert task.block_reason == "daily_cap"
    assert called == []


def test_upload_blocked_by_low_confidence_never_calls_tiktok(db_session, user, tmp_path, monkeypatch):
    task, _account = _seed_task(db_session, user, tmp_path, confidence_score=0.1)
    called = []
    monkeypatch.setattr(tiktok_client, "init_inbox_upload", lambda *a, **k: called.append("init"))

    upload.run(str(task.id))

    db_session.refresh(task)
    assert task.status == "blocked"
    assert task.block_reason == "low_confidence"
    assert called == []


def test_upload_missing_task_is_a_noop(db_session):
    upload.run(str(uuid.uuid4()))  # must not raise


def test_upload_fails_when_video_missing(db_session, user, tmp_path, monkeypatch):
    task, _account = _seed_task(db_session, user, tmp_path)
    with db_session.no_autoflush:
        clip = db_session.get(RenderedClip, task.rendered_clip_id)
        clip.object_key = None
        db_session.commit()

    upload.run(str(task.id))

    db_session.refresh(task)
    assert task.status == "failed"
    assert "missing" in task.last_error
