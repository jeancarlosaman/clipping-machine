"""Exercises app.workers.transcription.run() directly, against the real DB
and real object storage -- only the STT provider is faked, via
monkeypatching app.workers.transcription.get_stt_provider(), so these run
without a network call or a Whisper model download (faster-whisper's model
weights aren't reachable from this sandbox -- see test_local_whisper_provider.py
for what that would look like with real network access, and the README for
a one-time manual check on a real machine).
"""
import uuid

import pytest

from app.core.storage import storage
from app.core.stt.base import PermanentSttError, SttResult, SttSegment, TransientSttError
from app.db.models import StreamJob, Transcript
from app.workers import transcription
from app.workers.common import audio_object_key


class _FakeProvider:
    name = "fake"

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def transcribe(self, wav_path):
        if self._error:
            raise self._error
        return self._result


def _make_ingested_job(db_session, user, *, with_audio=True):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="ingested")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    if with_audio:
        # transcription.run() downloads via storage.get_local_path -- needs
        # a real (if fake-content) file at the deterministic audio key.
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav")
        with open(fd, "wb") as f:
            f.write(b"RIFF....WAVEfake")
        storage.put_file(path, audio_object_key(str(job.id)))
    return job


def test_transcription_success_path(db_session, user, monkeypatch):
    job = _make_ingested_job(db_session, user)

    fake_result = SttResult(
        segments=[SttSegment(start=0.0, end=1.5, text="hello world"), SttSegment(start=1.5, end=3.0, text="test")],
        full_text="hello world test",
        language="en",
    )
    monkeypatch.setattr(
        transcription, "get_stt_provider", lambda model_size=None: _FakeProvider(result=fake_result)
    )

    enqueued = []
    monkeypatch.setattr(
        "app.workers.transcription.enqueue", lambda stage, func, *a, **k: enqueued.append((stage, a))
    )

    transcription.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "transcribed"
    assert job.retry_count == 1
    assert enqueued == [("segmentation", (str(job.id),))]

    row = db_session.query(Transcript).filter_by(stream_job_id=job.id).one()
    assert row.provider == "fake"
    assert row.language == "en"
    assert row.full_text == "hello world test"
    assert row.segments == [
        {"start": 0.0, "end": 1.5, "text": "hello world"},
        {"start": 1.5, "end": 3.0, "text": "test"},
    ]


def test_transcription_permanent_failure_does_not_retry(db_session, user, monkeypatch):
    job = _make_ingested_job(db_session, user)
    monkeypatch.setattr(
        transcription, "get_stt_provider", lambda model_size=None: _FakeProvider(error=PermanentSttError("bad audio"))
    )

    # Should not raise -- a permanent failure is handled inside run(), not
    # left to bubble up and trigger a pointless RQ retry (mirrors ingest.py).
    transcription.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "failed_transcription"
    assert "bad audio" in job.last_error
    assert db_session.query(Transcript).filter_by(stream_job_id=job.id).count() == 0


def test_transcription_transient_failure_raises_for_rq_retry(db_session, user, monkeypatch):
    job = _make_ingested_job(db_session, user)
    monkeypatch.setattr(
        transcription, "get_stt_provider", lambda model_size=None: _FakeProvider(error=TransientSttError("rate limited"))
    )

    with pytest.raises(TransientSttError):
        transcription.run(str(job.id))

    db_session.refresh(job)
    # Still "transcribing", not yet failed -- RQ's Retry policy owns what
    # happens next; only on_failure() (after retries exhaust) marks it failed.
    assert job.status == "transcribing"


def test_transcription_missing_job_is_a_noop(db_session):
    # Should log and return quietly, not raise -- a stale/duplicate queue
    # message for a deleted job shouldn't crash the worker.
    transcription.run(str(uuid.uuid4()))


def test_on_failure_marks_job_failed(db_session, user):
    job = _make_ingested_job(db_session, user, with_audio=False)

    fake_rq_job = type("FakeRqJob", (), {"args": [str(job.id)]})()
    transcription.on_failure(fake_rq_job, None, RuntimeError, RuntimeError("exhausted"), None)

    db_session.refresh(job)
    assert job.status == "failed_transcription"
    assert "exhausted" in job.last_error
