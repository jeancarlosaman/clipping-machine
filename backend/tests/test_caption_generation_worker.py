"""Exercises app.workers.caption_generation.run() against a real DB, with
app.core.caption_generation.generate_caption_annotation monkeypatched so
these don't need network access -- that function's own behavior is covered
by test_caption_generation.py.
"""
import uuid

import app.workers.caption_generation as caption_worker
from app.db.models import CandidateSegment, RenderedClip, StreamJob, Transcript


def _make_clip(db_session, user, *, score_breakdown=None, caption_text="placeholder"):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="ready_for_review")
    db_session.add(job)
    db_session.flush()

    db_session.add(
        Transcript(
            stream_job_id=job.id,
            provider="fake",
            full_text="test",
            segments=[{"start": 0.0, "end": 2.0, "text": "wow insane"}],
        )
    )

    candidate = CandidateSegment(
        stream_job_id=job.id, start_seconds=0.0, end_seconds=2.0, status="selected",
        score_breakdown=score_breakdown,
    )
    db_session.add(candidate)
    db_session.flush()

    clip = RenderedClip(
        candidate_segment_id=candidate.id, stream_job_id=job.id, status="rendered", caption_text=caption_text,
    )
    db_session.add(clip)
    db_session.commit()
    db_session.refresh(candidate)
    db_session.refresh(clip)
    return candidate, clip


ANNOTATION = {
    "source": "llm",
    "reason": None,
    "hashtags": ["#insane", "#clutch"],
    "caption": "He went insane!",
    "explanation": "Huge reaction moment.",
    "model": "gpt-4o-mini",
    "generated_at": "2026-01-01T00:00:00+00:00",
}


def test_caption_generation_writes_annotation_and_updates_caption(db_session, user, monkeypatch):
    candidate, clip = _make_clip(db_session, user)
    monkeypatch.setattr(caption_worker, "generate_caption_annotation", lambda *a, **k: ANNOTATION)

    caption_worker.run(str(candidate.id))

    db_session.refresh(candidate)
    db_session.refresh(clip)
    assert candidate.llm_annotation == ANNOTATION
    assert clip.caption_text == "He went insane!"


def test_caption_generation_missing_candidate_is_a_noop(db_session):
    caption_worker.run(str(uuid.uuid4()))  # should not raise


def test_caption_generation_missing_clip_row_is_a_noop(db_session, user, monkeypatch):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="ready_for_review")
    db_session.add(job)
    db_session.flush()
    candidate = CandidateSegment(stream_job_id=job.id, start_seconds=0, end_seconds=2, status="selected")
    db_session.add(candidate)
    db_session.commit()
    db_session.refresh(candidate)

    called = []
    monkeypatch.setattr(caption_worker, "generate_caption_annotation", lambda *a, **k: called.append(1))

    caption_worker.run(str(candidate.id))  # no RenderedClip row -- should not raise

    assert called == []


def test_on_failure_logs_without_touching_db(db_session, user):
    candidate, clip = _make_clip(db_session, user)
    fake_rq_job = type("FakeRqJob", (), {"args": [str(candidate.id)]})()
    caption_worker.on_failure(fake_rq_job, None, RuntimeError, RuntimeError("boom"), None)  # should not raise

    db_session.refresh(clip)
    assert clip.caption_text == "placeholder"  # untouched
