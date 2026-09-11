import uuid

from app.core.auth import create_access_token
from app.core.storage import storage
from app.db.models import CandidateSegment, CreatorAccount, RenderedClip, StreamJob, User


def _seed_clip(db_session, user):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="ready_for_review")
    db_session.add(job)
    db_session.flush()

    segment = CandidateSegment(stream_job_id=job.id, start_seconds=0, end_seconds=10, status="selected")
    db_session.add(segment)
    db_session.flush()

    clip = RenderedClip(candidate_segment_id=segment.id, stream_job_id=job.id, status="rendered")
    db_session.add(clip)

    account = CreatorAccount(
        user_id=user.id,
        platform="tiktok",
        external_account_id="acct-1",
        access_token_encrypted="token",
    )
    db_session.add(account)
    db_session.commit()
    db_session.refresh(clip)
    db_session.refresh(account)
    return clip, account


def test_upload_blocked_without_approval(client, auth_headers, db_session, user, monkeypatch):
    clip, account = _seed_clip(db_session, user)
    monkeypatch.setattr("app.api.routers.clips.enqueue", lambda *a, **k: None)

    resp = client.post(
        f"/api/v1/clips/{clip.id}/upload",
        headers=auth_headers,
        json={"creator_account_id": str(account.id)},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "review_required"


def test_approve_then_upload_succeeds(client, auth_headers, db_session, user, monkeypatch):
    clip, account = _seed_clip(db_session, user)
    enqueued = []
    monkeypatch.setattr(
        "app.api.routers.clips.enqueue", lambda stage, func, *a, **k: enqueued.append((stage, a))
    )

    review_resp = client.post(
        f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "approved"}
    )
    assert review_resp.status_code == 201
    assert review_resp.json()["decision"] == "approved"

    upload_resp = client.post(
        f"/api/v1/clips/{clip.id}/upload",
        headers=auth_headers,
        json={"creator_account_id": str(account.id)},
    )
    assert upload_resp.status_code == 202
    body = upload_resp.json()
    assert body["status"] == "queued"
    assert body["target_mode"] == "draft"
    assert enqueued == [("upload", (body["id"],))]


def test_upload_rejects_direct_target_mode(client, auth_headers, db_session, user):
    clip, account = _seed_clip(db_session, user)
    client.post(f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "approved"})

    resp = client.post(
        f"/api/v1/clips/{clip.id}/upload",
        headers=auth_headers,
        json={"creator_account_id": str(account.id), "target_mode": "direct"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_review_with_rating_is_stored_and_returned(client, auth_headers, db_session, user):
    clip, _account = _seed_clip(db_session, user)

    resp = client.post(
        f"/api/v1/clips/{clip.id}/review",
        headers=auth_headers,
        json={"decision": "approved", "rating": 4},
    )
    assert resp.status_code == 201
    assert resp.json()["rating"] == 4


def test_review_rating_out_of_range_rejected(client, auth_headers, db_session, user):
    clip, _account = _seed_clip(db_session, user)

    resp = client.post(
        f"/api/v1/clips/{clip.id}/review",
        headers=auth_headers,
        json={"decision": "approved", "rating": 6},
    )
    assert resp.status_code == 422

    resp = client.post(
        f"/api/v1/clips/{clip.id}/review",
        headers=auth_headers,
        json={"decision": "approved", "rating": 0},
    )
    assert resp.status_code == 422


def test_clip_exposes_latest_rating_across_multiple_reviews(client, auth_headers, db_session, user):
    clip, _account = _seed_clip(db_session, user)

    client.post(f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "approved", "rating": 2})
    client.post(f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "approved", "rating": 5})

    db_session.refresh(clip)
    assert clip.latest_rating == 5


def test_clip_latest_rating_ignores_unrated_review_after_a_rated_one(client, auth_headers, db_session, user):
    # A reviewer re-reviewing without a rating shouldn't erase the earlier
    # rating -- latest_rating means "most recent review that HAD a rating,"
    # not "the rating field on the single most recent review row."
    clip, _account = _seed_clip(db_session, user)

    client.post(f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "approved", "rating": 3})
    client.post(f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "skipped"})

    db_session.refresh(clip)
    assert clip.latest_rating == 3


def test_clip_exposes_latest_notes_across_multiple_reviews(client, auth_headers, db_session, user):
    clip, _account = _seed_clip(db_session, user)

    client.post(
        f"/api/v1/clips/{clip.id}/review",
        headers=auth_headers,
        json={"decision": "approved", "notes": "good story but starts mid-sentence"},
    )
    client.post(
        f"/api/v1/clips/{clip.id}/review",
        headers=auth_headers,
        json={"decision": "approved", "notes": "actually the refund rant is the real hook here"},
    )

    db_session.refresh(clip)
    assert clip.latest_notes == "actually the refund rant is the real hook here"


def test_clip_latest_notes_ignores_uncommented_review_after_a_commented_one(
    client, auth_headers, db_session, user
):
    # Same semantics as latest_rating: re-reviewing with an empty comment
    # box means "no comment this time," not "erase what I said before."
    clip, _account = _seed_clip(db_session, user)

    client.post(
        f"/api/v1/clips/{clip.id}/review",
        headers=auth_headers,
        json={"decision": "approved", "notes": "the callback at the end makes it"},
    )
    client.post(f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "skipped"})

    db_session.refresh(clip)
    assert clip.latest_notes == "the callback at the end makes it"


def test_clip_latest_notes_is_none_when_never_commented(client, auth_headers, db_session, user):
    clip, _account = _seed_clip(db_session, user)
    client.post(f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "approved"})

    db_session.refresh(clip)
    assert clip.latest_notes is None


def test_review_nonexistent_clip_404s(client, auth_headers):
    resp = client.post(
        f"/api/v1/clips/{uuid.uuid4()}/review", headers=auth_headers, json={"decision": "approved"}
    )
    assert resp.status_code == 404


def test_reject_then_delete_clip_removes_it_and_its_files(client, auth_headers, db_session, user, tmp_path):
    clip, _account = _seed_clip(db_session, user)
    video_key, thumb_key = "clips/x/video.mp4", "clips/x/thumb.jpg"
    video_src = tmp_path / "v.mp4"
    thumb_src = tmp_path / "t.jpg"
    video_src.write_bytes(b"fake video bytes")
    thumb_src.write_bytes(b"fake thumb bytes")
    storage.put_file(str(video_src), video_key)
    storage.put_file(str(thumb_src), thumb_key)
    clip.object_key = video_key
    clip.thumbnail_key = thumb_key
    db_session.commit()

    reject_resp = client.post(
        f"/api/v1/clips/{clip.id}/review", headers=auth_headers, json={"decision": "rejected"}
    )
    assert reject_resp.status_code == 201

    delete_resp = client.delete(f"/api/v1/clips/{clip.id}", headers=auth_headers)
    assert delete_resp.status_code == 204

    # The delete happened through a different DB session (the API request's
    # own). expire_all() would still raise ObjectDeletedError on access
    # (SQLAlchemy's documented behavior for an expired-but-tracked identity
    # whose row is actually gone) -- expunge_all() fully detaches so the
    # next .get() is a plain fresh query, correctly returning None.
    db_session.expunge_all()
    assert db_session.get(RenderedClip, clip.id) is None
    assert not storage.exists(video_key)
    assert not storage.exists(thumb_key)


def test_delete_clip_enforces_ownership(client, db_session, user):
    other_user = User(email=f"{uuid.uuid4()}@example.com", password_hash="x")
    db_session.add(other_user)
    db_session.commit()
    db_session.refresh(other_user)
    clip, _account = _seed_clip(db_session, other_user)

    my_headers = {"Authorization": f"Bearer {create_access_token(str(user.id))}"}
    resp = client.delete(f"/api/v1/clips/{clip.id}", headers=my_headers)
    assert resp.status_code == 403
    assert db_session.get(RenderedClip, clip.id) is not None


def test_delete_nonexistent_clip_404s(client, auth_headers):
    resp = client.delete(f"/api/v1/clips/{uuid.uuid4()}", headers=auth_headers)
    assert resp.status_code == 404
