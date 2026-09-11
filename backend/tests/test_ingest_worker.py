"""Exercises app.workers.ingest.run() directly (not through the queue) --
real ffprobe/ffmpeg calls against a real generated video, real local storage.
This is the one worker with real logic in the first pass; the others are
covered by test_stub_workers.py.
"""
from app.core.storage import storage
from app.db.models import StreamJob
from app.workers import ingest
from app.workers.common import audio_object_key


def test_ingest_success_path(db_session, user, sample_video_path, monkeypatch):
    object_key = "raw/sample.mp4"
    storage.put_file(str(sample_video_path), object_key)

    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key=object_key, status="queued")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    enqueued = []
    monkeypatch.setattr(
        "app.workers.ingest.enqueue", lambda stage, func, *a, **k: enqueued.append((stage, a))
    )

    ingest.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "ingested"
    assert job.duration_seconds is not None
    assert job.duration_seconds > 1.0
    assert job.retry_count == 1
    assert storage.exists(audio_object_key(str(job.id)))
    assert enqueued == [("transcription", (str(job.id),))]


def test_ingest_permanent_failure_on_corrupt_file(db_session, user):
    object_key = "raw/corrupt.mp4"
    storage.put_file(_write_garbage_file(), object_key)

    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key=object_key, status="queued")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    # Should not raise -- permanent failures are handled inside run(), not
    # left to bubble up and trigger a pointless RQ retry (see ingest.py).
    ingest.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "failed_ingest"
    assert job.last_error


def test_ingest_missing_job_is_a_noop(db_session):
    import uuid

    # Should log and return quietly, not raise -- a stale/duplicate queue
    # message for a deleted job shouldn't crash the worker.
    ingest.run(str(uuid.uuid4()))


def _write_garbage_file() -> str:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".mp4")
    with open(fd, "wb") as f:
        f.write(b"this is not a real video file")
    return path
