"""Exercises app.workers.segmentation.run() directly -- real DB, real
storage, real PySceneDetect against a real generated video. Only the
transcript is fabricated (no need for real STT here; that's already
covered by test_transcription_worker.py).
"""
import uuid

from app.core.segmentation_logic import detect_scene_cuts as real_detect_scene_cuts
from app.core.storage import storage
from app.db.models import CandidateSegment, StreamJob, Transcript
from app.workers import segmentation


def _make_job_with_transcript(db_session, user, video_path, segments):
    object_key = "raw/segmentation-test.mp4"
    storage.put_file(str(video_path), object_key)

    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key=object_key, status="transcribed")
    db_session.add(job)
    db_session.flush()

    db_session.add(
        Transcript(stream_job_id=job.id, provider="fake", full_text="test", segments=segments)
    )
    db_session.commit()
    db_session.refresh(job)
    return job


def test_segmentation_success_creates_candidate_segments(db_session, user, three_scene_video_path, monkeypatch):
    # Real 9s video, real scene cuts at ~3s/6s. Lower min/max so the whole
    # thing qualifies as candidates without needing a much longer fixture.
    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 2.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 4.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_silence_gap_seconds", 1.0)

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 9.0, "text": "talking the whole time"}]
    )

    enqueued = []
    monkeypatch.setattr(
        "app.workers.segmentation.enqueue", lambda stage, func, *a, **k: enqueued.append((stage, a))
    )

    segmentation.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "segmented"
    assert job.retry_count == 1
    assert enqueued == [("scoring", (str(job.id),))]

    segments = (
        db_session.query(CandidateSegment)
        .filter_by(stream_job_id=job.id)
        .order_by(CandidateSegment.start_seconds)
        .all()
    )
    # More than one candidate for this block -- segmentation now slides
    # overlapping windows across a too-long block instead of committing to
    # one rigid partition (see app.core.segmentation_logic._sliding_subwindows);
    # scoring.py's non-max suppression is what turns these back into a
    # diverse final selection, not segmentation itself.
    assert len(segments) > 1
    assert all(s.status == "pending_score" for s in segments)
    assert float(segments[0].start_seconds) == 0.0
    assert float(segments[-1].end_seconds) == 9.0
    # At least one window boundary should land near the real detected scene
    # changes (3.0/6.0), not only at arbitrary hard-cut points -- loose
    # tolerance since exact detection timing can vary slightly by
    # PySceneDetect/OpenCV version.
    boundaries = [float(v) for s in segments for v in (s.start_seconds, s.end_seconds)]
    assert any(2.5 < b < 3.5 for b in boundaries)
    assert any(5.5 < b < 6.5 for b in boundaries)

    # The scene cuts segmentation detected are persisted on the job so
    # scoring.py can reuse them instead of re-detecting -- see
    # StreamJob.scene_cuts_seconds.
    assert job.scene_cuts_seconds is not None
    assert len(job.scene_cuts_seconds) == 2

    # ENABLE_LLM_SEGMENT_SUGGESTIONS defaults to False -- every candidate
    # here should be a plain heuristic one, with no llm_reason.
    assert all(s.origin == "heuristic" for s in segments)
    assert all(s.llm_reason is None for s in segments)


def test_segmentation_adds_llm_suggested_candidates_alongside_heuristic_ones(
    db_session, user, three_scene_video_path, monkeypatch
):
    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 2.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 4.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_silence_gap_seconds", 1.0)
    monkeypatch.setattr("app.workers.segmentation.settings.enable_llm_segment_suggestions", True)

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 9.0, "text": "talking the whole time"}]
    )
    monkeypatch.setattr("app.workers.segmentation.enqueue", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.workers.segmentation.generate_llm_segment_suggestions",
        lambda transcript_segments, min_len, max_len, feedback_examples=None: [
            {"start": 1.0, "end": 3.5, "reason": "A big reaction."}
        ],
    )

    segmentation.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "segmented"

    segments = db_session.query(CandidateSegment).filter_by(stream_job_id=job.id).all()
    heuristic = [s for s in segments if s.origin == "heuristic"]
    llm = [s for s in segments if s.origin == "llm"]
    assert len(heuristic) >= 1  # the same sliding-window candidates as the plain test above
    assert len(llm) == 1
    assert float(llm[0].start_seconds) == 1.0
    assert float(llm[0].end_seconds) == 3.5
    assert llm[0].llm_reason == "A big reaction."
    assert all(s.status == "pending_score" for s in segments)  # same pool, same starting status


def test_segmentation_llm_suggestions_can_rescue_a_job_with_no_heuristic_windows(
    db_session, user, three_scene_video_path, monkeypatch
):
    # Same premise as test_segmentation_no_windows_fails_clearly (a 5s
    # transcript block below a 15s min_len produces zero heuristic
    # candidates) -- but with an LLM suggestion available, the job should
    # succeed using that suggestion alone instead of failing outright. This
    # is the one behavior change this feature makes to the pre-existing
    # "no windows" path, and it's opt-in (only reachable when the flag is
    # explicitly turned on).
    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 15.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 90.0)
    monkeypatch.setattr("app.workers.segmentation.settings.enable_llm_segment_suggestions", True)

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 5.0, "text": "too short"}]
    )
    monkeypatch.setattr("app.workers.segmentation.enqueue", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.workers.segmentation.generate_llm_segment_suggestions",
        lambda transcript_segments, min_len, max_len, feedback_examples=None: [
            {"start": 0.0, "end": 5.0, "reason": "Rescued by LLM."}
        ],
    )

    segmentation.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "segmented"

    segments = db_session.query(CandidateSegment).filter_by(stream_job_id=job.id).all()
    assert len(segments) == 1
    assert segments[0].origin == "llm"
    assert segments[0].llm_reason == "Rescued by LLM."


def test_segmentation_llm_suggestion_failure_does_not_fail_the_job(
    db_session, user, three_scene_video_path, monkeypatch
):
    # Defense in depth: generate_llm_segment_suggestions is documented to
    # never raise, but segmentation's own success must not depend on that
    # promise holding -- a broken opt-in extra signal should degrade to
    # zero LLM candidates, not sink an otherwise-successful job.
    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 2.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 4.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_silence_gap_seconds", 1.0)
    monkeypatch.setattr("app.workers.segmentation.settings.enable_llm_segment_suggestions", True)

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 9.0, "text": "talking the whole time"}]
    )
    monkeypatch.setattr("app.workers.segmentation.enqueue", lambda *a, **k: None)

    def _boom(transcript_segments, min_len, max_len, feedback_examples=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.workers.segmentation.generate_llm_segment_suggestions", _boom)

    segmentation.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "segmented"
    segments = db_session.query(CandidateSegment).filter_by(stream_job_id=job.id).all()
    assert len(segments) >= 1
    assert all(s.origin == "heuristic" for s in segments)


def test_segmentation_passes_configured_frame_skip_to_scene_detection(
    db_session, user, three_scene_video_path, monkeypatch
):
    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 2.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 4.0)
    monkeypatch.setattr("app.workers.segmentation.settings.scene_detect_frame_skip", 2)

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 9.0, "text": "talking the whole time"}]
    )
    monkeypatch.setattr("app.workers.segmentation.enqueue", lambda *a, **k: None)

    calls = []

    def spy(video_path, frame_skip=0):
        calls.append(frame_skip)
        return real_detect_scene_cuts(video_path, frame_skip=frame_skip)

    monkeypatch.setattr("app.workers.segmentation.detect_scene_cuts", spy)

    segmentation.run(str(job.id))

    assert calls == [2]


def test_segmentation_no_transcript_fails_permanently(db_session, user):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="transcribed")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    segmentation.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "failed_segmentation"
    assert "no transcript" in job.last_error


def test_segmentation_rejects_inverted_min_max_clip_length(db_session, user, three_scene_video_path, monkeypatch):
    # Defense in depth for the same real bug covered by
    # test_stream_jobs_api.py's effective-min/max tests: this worker has no
    # guarantee every job it processes came through that endpoint's
    # validation (an old row from before that check existed, a manual DB
    # edit, a future caller) -- with min_len > max_len,
    # build_candidate_windows/_sliding_subwindows has no guard of its own
    # and silently produces ~max_len-long candidates spaced ~min_len apart
    # instead of failing, which is exactly the "clips are way shorter than
    # my configured minimum" bug a user hit for real. Segmentation should
    # refuse outright instead.
    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 35.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 10.0)

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 9.0, "text": "talking the whole time"}]
    )

    segmentation.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "failed_segmentation"
    assert "min_clip_seconds" in job.last_error
    assert "max_clip_seconds" in job.last_error
    assert db_session.query(CandidateSegment).filter_by(stream_job_id=job.id).count() == 0


def test_segmentation_no_windows_fails_clearly(db_session, user, three_scene_video_path, monkeypatch):
    # Explicit min_len rather than relying on the config default -- local
    # dev .env files commonly lower SEGMENT_MIN_CLIP_SECONDS for short test
    # videos (see README), which would otherwise make this test's premise
    # (a 5s segment is too short) silently false depending on environment.
    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 15.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 90.0)

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 5.0, "text": "too short"}]
    )

    segmentation.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "failed_segmentation"
    assert "no candidate windows" in job.last_error


def test_segmentation_missing_job_is_a_noop(db_session):
    segmentation.run(str(uuid.uuid4()))


def test_on_failure_marks_job_failed(db_session, user):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="transcribed")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    fake_rq_job = type("FakeRqJob", (), {"args": [str(job.id)]})()
    segmentation.on_failure(fake_rq_job, None, RuntimeError, RuntimeError("exhausted"), None)

    db_session.refresh(job)
    assert job.status == "failed_segmentation"
    assert "exhausted" in job.last_error


def test_segmentation_passes_reviewer_comments_as_feedback_examples(
    db_session, user, three_scene_video_path, monkeypatch
):
    # The "learn from my feedback" loop: a past clip this creator reviewed
    # WITH a written comment should reach the LLM proposer as a few-shot
    # example, carrying the decision, the comment, and the real transcript
    # text of the window they were judging.
    from app.db.models import RenderedClip, ReviewDecision

    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 2.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 4.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_silence_gap_seconds", 1.0)
    monkeypatch.setattr("app.workers.segmentation.settings.enable_llm_segment_suggestions", True)

    # An older, already-reviewed job from the same creator.
    old_job = _make_job_with_transcript(
        db_session, user, three_scene_video_path,
        [{"start": 0.0, "end": 4.0, "text": "the part about the refund policy"}],
    )
    old_candidate = CandidateSegment(
        stream_job_id=old_job.id, start_seconds=0.0, end_seconds=4.0, status="selected"
    )
    db_session.add(old_candidate)
    db_session.flush()
    old_clip = RenderedClip(
        candidate_segment_id=old_candidate.id, stream_job_id=old_job.id, status="rendered"
    )
    db_session.add(old_clip)
    db_session.flush()
    db_session.add(
        ReviewDecision(
            rendered_clip_id=old_clip.id,
            user_id=user.id,
            decision="approved",
            notes="the refund policy tangent is the actual story here",
        )
    )
    db_session.commit()

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 9.0, "text": "talking the whole time"}]
    )
    monkeypatch.setattr("app.workers.segmentation.enqueue", lambda *a, **k: None)

    seen = {}

    def _capture(transcript_segments, min_len, max_len, feedback_examples=None):
        seen["examples"] = feedback_examples
        return []

    monkeypatch.setattr("app.workers.segmentation.generate_llm_segment_suggestions", _capture)

    segmentation.run(str(job.id))

    examples = seen["examples"]
    assert len(examples) == 1
    assert examples[0]["decision"] == "approved"
    assert examples[0]["notes"] == "the refund policy tangent is the actual story here"
    # Transcript excerpt comes from the reviewed clip's own window, not
    # from a (possibly LLM-rewritten) stored caption.
    assert "refund policy" in examples[0]["transcript"]


def test_segmentation_skips_reviews_without_a_written_comment(
    db_session, user, three_scene_video_path, monkeypatch
):
    # An approve/reject with no reasoning teaches the few-shot block
    # nothing -- it should never become an example.
    from app.db.models import RenderedClip, ReviewDecision

    monkeypatch.setattr("app.workers.segmentation.settings.segment_min_clip_seconds", 2.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_max_clip_seconds", 4.0)
    monkeypatch.setattr("app.workers.segmentation.settings.segment_silence_gap_seconds", 1.0)
    monkeypatch.setattr("app.workers.segmentation.settings.enable_llm_segment_suggestions", True)

    old_job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 4.0, "text": "whatever"}]
    )
    old_candidate = CandidateSegment(
        stream_job_id=old_job.id, start_seconds=0.0, end_seconds=4.0, status="selected"
    )
    db_session.add(old_candidate)
    db_session.flush()
    old_clip = RenderedClip(
        candidate_segment_id=old_candidate.id, stream_job_id=old_job.id, status="rendered"
    )
    db_session.add(old_clip)
    db_session.flush()
    db_session.add(
        ReviewDecision(rendered_clip_id=old_clip.id, user_id=user.id, decision="approved", notes=None)
    )
    db_session.commit()

    job = _make_job_with_transcript(
        db_session, user, three_scene_video_path, [{"start": 0.0, "end": 9.0, "text": "talking the whole time"}]
    )
    monkeypatch.setattr("app.workers.segmentation.enqueue", lambda *a, **k: None)

    seen = {}

    def _capture(transcript_segments, min_len, max_len, feedback_examples=None):
        seen["examples"] = feedback_examples
        return []

    monkeypatch.setattr("app.workers.segmentation.generate_llm_segment_suggestions", _capture)

    segmentation.run(str(job.id))

    assert seen["examples"] == []
