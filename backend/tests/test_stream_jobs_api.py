import uuid

from app.core.auth import create_access_token
from app.core.storage import storage
from app.db.models import CandidateSegment, RenderedClip, StreamJob, User


def test_create_stream_job_rejects_unsupported_extension(client, auth_headers, tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not a video")

    resp = client.post(
        "/api/v1/stream-jobs",
        headers=auth_headers,
        files={"file": ("notes.txt", bad_file.open("rb"), "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_file_type"


def test_create_stream_job_from_upload_enqueues_ingest(client, auth_headers, sample_video_path, monkeypatch):
    enqueued = {}

    def fake_enqueue(stage, func, *args, **kwargs):
        enqueued["stage"] = stage
        enqueued["args"] = args
        return None

    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", fake_enqueue)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["source_type"] == "upload"
    assert enqueued["stage"] == "ingest"
    assert enqueued["args"] == (body["id"],)


def test_create_stream_job_accepts_valid_overrides(client, auth_headers, sample_video_path, monkeypatch):
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={
                "max_clips": "5",
                "min_score_threshold": "4.5",
                "min_clip_seconds": "10",
                "max_clip_seconds": "60",
                "stt_model_size": "tiny",
            },
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["max_clips"] == 5
    assert body["min_score_threshold"] == 4.5
    assert body["min_clip_seconds"] == 10.0
    assert body["max_clip_seconds"] == 60.0
    assert body["stt_model_size"] == "tiny"


def test_create_stream_job_accepts_camera_layout_mode_override(client, auth_headers, sample_video_path, monkeypatch):
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"camera_layout_mode": "single_crop"},
        )

    assert resp.status_code == 201
    assert resp.json()["camera_layout_mode"] == "single_crop"


def test_create_stream_job_camera_layout_mode_auto_stores_null(client, auth_headers, sample_video_path, monkeypatch):
    # "auto" is accepted as an explicit input alias for "use the default
    # heuristic" -- confirms it's normalized to null, same as not sending
    # the field at all, rather than being stored as the literal string
    # "auto" (app/workers/rendering.py only checks for "single_crop" /
    # "split_reaction" and treats anything else, including a stray "auto",
    # as the default path -- but storing null is the intended contract).
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"camera_layout_mode": "auto"},
        )

    assert resp.status_code == 201
    assert resp.json()["camera_layout_mode"] is None


def test_create_stream_job_rejects_invalid_camera_layout_mode(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"camera_layout_mode": "always_split_please"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_camera_layout_mode"


def test_create_stream_job_without_overrides_uses_defaults(client, auth_headers, sample_video_path, monkeypatch):
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs", headers=auth_headers, files={"file": ("sample.mp4", f, "video/mp4")}
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["max_clips"] == 10  # settings.default_max_clips_per_job
    assert body["min_score_threshold"] is None
    assert body["min_clip_seconds"] is None
    assert body["max_clip_seconds"] is None
    assert body["stt_model_size"] is None
    assert body["camera_layout_mode"] is None


def test_create_stream_job_rejects_max_clips_above_hard_cap(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"max_clips": "999"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_max_clips"


def test_create_stream_job_rejects_max_clips_below_one(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"max_clips": "0"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_max_clips"


def test_create_stream_job_rejects_min_score_threshold_out_of_range(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"min_score_threshold": "11"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_min_score_threshold"


def test_create_stream_job_rejects_min_clip_seconds_below_floor(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"min_clip_seconds": "1"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_min_clip_seconds"


def test_create_stream_job_rejects_max_clip_seconds_above_ceiling(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"max_clip_seconds": "999"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_max_clip_seconds"


def test_create_stream_job_rejects_min_greater_than_max_clip_length(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"min_clip_seconds": "50", "max_clip_seconds": "40"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_clip_length_range"


def test_create_stream_job_rejects_min_clip_seconds_above_effective_max_default(
    client, auth_headers, sample_video_path, monkeypatch
):
    # Regression test for a real production bug: a job explicitly overrode
    # min_clip_seconds=35 while leaving max_clip_seconds unset (so it fell
    # back to a .env SEGMENT_MAX_CLIP_SECONDS that had been left at 10 from
    # earlier short-test-video tuning). The old validation only compared
    # min/max when BOTH were provided on the same request, so this
    # combination sailed through -- and inverted min>max downstream silently
    # broke segmentation_logic.build_candidate_windows/_sliding_subwindows
    # into producing ~max_len-long candidates (~10s) spaced ~min_len apart
    # (~35s), a real "all my clips are 9-10 seconds" bug a user hit despite
    # explicitly setting a 35s minimum. This must now be rejected at
    # request time instead.
    monkeypatch.setattr("app.api.routers.stream_jobs.settings.segment_max_clip_seconds", 10.0)
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"min_clip_seconds": "35"},  # max_clip_seconds left unset -- effective max is 10.0 above
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_clip_length_range"


def test_create_stream_job_rejects_max_clip_seconds_below_effective_min_default(
    client, auth_headers, sample_video_path, monkeypatch
):
    # Mirror case: max_clip_seconds explicitly set, min_clip_seconds left to
    # fall back to a .env default that's larger than it.
    monkeypatch.setattr("app.api.routers.stream_jobs.settings.segment_min_clip_seconds", 35.0)
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"max_clip_seconds": "10"},  # min_clip_seconds left unset -- effective min is 35.0 above
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_clip_length_range"


def test_create_stream_job_rejects_unknown_stt_model_size(client, auth_headers, sample_video_path):
    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"stt_model_size": "gigantic"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_stt_model_size"


def test_get_stream_job_enforces_ownership(client, db_session, user):
    other_user = User(email=f"{uuid.uuid4()}@example.com", password_hash="x")
    db_session.add(other_user)
    db_session.commit()
    db_session.refresh(other_user)

    job = StreamJob(user_id=other_user.id, source_type="upload", raw_object_key="raw/x.mp4")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    my_headers = {"Authorization": f"Bearer {create_access_token(str(user.id))}"}
    resp = client.get(f"/api/v1/stream-jobs/{job.id}", headers=my_headers)
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_get_nonexistent_stream_job_404s(client, auth_headers):
    resp = client.get(f"/api/v1/stream-jobs/{uuid.uuid4()}", headers=auth_headers)
    assert resp.status_code == 404


def test_delete_stream_job_removes_row_and_files(client, auth_headers, db_session, user, tmp_path):
    raw_key = "raw/delete-test.mp4"
    clip_video_key, clip_thumb_key = "clips/delete-test/v.mp4", "clips/delete-test/t.jpg"
    for key in (raw_key, clip_video_key, clip_thumb_key):
        src = tmp_path / key.replace("/", "_")
        src.write_bytes(b"fake bytes")
        storage.put_file(str(src), key)

    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key=raw_key, status="scored")
    db_session.add(job)
    db_session.flush()
    segment = CandidateSegment(stream_job_id=job.id, start_seconds=0, end_seconds=5, status="selected")
    db_session.add(segment)
    db_session.flush()
    clip = RenderedClip(
        candidate_segment_id=segment.id, stream_job_id=job.id, status="rendered",
        object_key=clip_video_key, thumbnail_key=clip_thumb_key,
    )
    db_session.add(clip)
    db_session.commit()
    job_id = job.id
    clip_id = clip.id  # captured now -- clip becomes detached below

    resp = client.delete(f"/api/v1/stream-jobs/{job_id}", headers=auth_headers)
    assert resp.status_code == 204

    # The delete happened through a different DB session (the API request's
    # own). expire_all() would still raise ObjectDeletedError on access
    # (SQLAlchemy's documented behavior for an expired-but-tracked identity
    # whose row is actually gone) -- expunge_all() fully detaches so the
    # next .get() is a plain fresh query, correctly returning None.
    db_session.expunge_all()
    assert db_session.get(StreamJob, job_id) is None
    assert db_session.get(RenderedClip, clip_id) is None  # cascaded
    assert not storage.exists(raw_key)
    assert not storage.exists(clip_video_key)
    assert not storage.exists(clip_thumb_key)


def test_delete_stream_job_enforces_ownership(client, db_session, user):
    other_user = User(email=f"{uuid.uuid4()}@example.com", password_hash="x")
    db_session.add(other_user)
    db_session.commit()
    db_session.refresh(other_user)

    job = StreamJob(user_id=other_user.id, source_type="upload", raw_object_key="raw/x.mp4")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    my_headers = {"Authorization": f"Bearer {create_access_token(str(user.id))}"}
    resp = client.delete(f"/api/v1/stream-jobs/{job.id}", headers=my_headers)
    assert resp.status_code == 403
    assert db_session.get(StreamJob, job.id) is not None


def test_delete_nonexistent_stream_job_404s(client, auth_headers):
    resp = client.delete(f"/api/v1/stream-jobs/{uuid.uuid4()}", headers=auth_headers)
    assert resp.status_code == 404


def test_create_stream_job_accepts_fit_frame_camera_layout_mode(
    client, auth_headers, sample_video_path, monkeypatch
):
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"camera_layout_mode": "fit_frame"},
        )

    assert resp.status_code == 201
    assert resp.json()["camera_layout_mode"] == "fit_frame"


def test_create_stream_job_accepts_crop_bias_override(
    client, auth_headers, sample_video_path, monkeypatch
):
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"crop_bias": "left"},
        )

    assert resp.status_code == 201
    assert resp.json()["crop_bias"] == "left"


def test_create_stream_job_rejects_invalid_crop_bias(
    client, auth_headers, sample_video_path, monkeypatch
):
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"crop_bias": "diagonal"},
        )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_crop_bias"


def test_create_stream_job_crop_bias_center_is_stored_not_treated_as_a_default(
    client, auth_headers, sample_video_path, monkeypatch
):
    # Unlike camera_layout_mode's "auto", an explicit "center" is a real
    # instruction ("center it and ignore faces"), so it must persist rather
    # than collapsing to null.
    monkeypatch.setattr("app.api.routers.stream_jobs.enqueue", lambda *a, **k: None)

    with sample_video_path.open("rb") as f:
        resp = client.post(
            "/api/v1/stream-jobs",
            headers=auth_headers,
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"crop_bias": "center"},
        )

    assert resp.status_code == 201
    assert resp.json()["crop_bias"] == "center"
