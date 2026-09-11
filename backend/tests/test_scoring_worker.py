"""Exercises app.workers.scoring.run() directly -- real DB, real storage,
real PySceneDetect against a real generated video (same fixture as
test_segmentation_worker.py). Only the transcript/candidates are fabricated
-- no need to run real STT/segmentation here, that's already covered
elsewhere.
"""
import uuid

from app.core.storage import storage
from app.db.models import CandidateSegment, RenderedClip, StreamJob, Transcript
from app.workers import scoring

TRANSCRIPT_SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "This is insane, wow!"},
    {"start": 2.5, "end": 4.0, "text": "Can you believe that?"},
    {"start": 4.1, "end": 6.0, "text": "That was hilarious and unreal."},
    {"start": 6.5, "end": 9.0, "text": "yeah okay sure whatever"},
]


def _make_job(db_session, user, video_path, *, max_clips=10):
    object_key = "raw/scoring-test.mp4"
    storage.put_file(str(video_path), object_key)

    job = StreamJob(
        user_id=user.id, source_type="upload", raw_object_key=object_key,
        status="segmented", max_clips=max_clips,
    )
    db_session.add(job)
    db_session.flush()

    db_session.add(
        Transcript(stream_job_id=job.id, provider="fake", full_text="test", segments=TRANSCRIPT_SEGMENTS)
    )
    db_session.commit()
    db_session.refresh(job)
    return job


def _add_candidates(db_session, job, windows):
    segments = []
    for start, end in windows:
        segment = CandidateSegment(stream_job_id=job.id, start_seconds=start, end_seconds=end, status="pending_score")
        db_session.add(segment)
        segments.append(segment)
    db_session.commit()
    for s in segments:
        db_session.refresh(s)
    return segments


def test_scoring_success_selects_top_clips_and_enqueues_rendering(db_session, user, three_scene_video_path, monkeypatch):
    job = _make_job(db_session, user, three_scene_video_path, max_clips=1)
    # Two windows: [0,6] has lots of emotional/question signal, [6.5,9] is flat.
    segments = _add_candidates(db_session, job, [(0.0, 6.0), (6.5, 9.0)])

    enqueued = []
    monkeypatch.setattr(
        "app.workers.scoring.enqueue",
        lambda stage, func, *a, **k: enqueued.append((stage, a, k)),
    )

    scoring.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "scored"
    assert job.retry_count == 1

    db_session.refresh(segments[0])
    db_session.refresh(segments[1])
    assert segments[0].status == "selected"  # higher-signal window
    assert segments[1].status == "rejected"
    assert segments[0].score is not None
    assert segments[0].score_breakdown is not None
    assert segments[0].features is not None
    assert float(segments[0].score) > float(segments[1].score)

    clips = db_session.query(RenderedClip).filter_by(stream_job_id=job.id).all()
    assert len(clips) == 1
    assert clips[0].candidate_segment_id == segments[0].id
    assert clips[0].status == "pending"
    assert clips[0].caption_text  # non-empty default caption

    assert len(enqueued) == 1
    stage, args, kwargs = enqueued[0]
    assert stage == "rendering"
    assert args == (str(segments[0].id),)
    assert kwargs["max_retries"] == 3


def test_scoring_reuses_persisted_scene_cuts_and_skips_redetection(
    db_session, user, three_scene_video_path, monkeypatch
):
    job = _make_job(db_session, user, three_scene_video_path, max_clips=1)
    job.scene_cuts_seconds = [1.0, 3.0]  # segmentation.py would have persisted this
    db_session.commit()
    segments = _add_candidates(db_session, job, [(0.0, 6.0)])

    def _fail_if_called(path):
        raise AssertionError("detect_scene_cuts should not be called when scene_cuts_seconds is cached")

    monkeypatch.setattr("app.workers.scoring.detect_scene_cuts", _fail_if_called)
    monkeypatch.setattr("app.workers.scoring.enqueue", lambda *a, **k: None)

    scoring.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "scored"
    db_session.refresh(segments[0])
    # The cached cuts (both inside [0, 6]) should have been used for the
    # scene_changes feature, proving they actually got fed into scoring, not
    # just silently ignored.
    assert segments[0].features["scene_changes"] == 2.0


def test_scoring_min_score_threshold_can_reject_everything(db_session, user, three_scene_video_path, monkeypatch):
    job = _make_job(db_session, user, three_scene_video_path, max_clips=5)
    job.min_score_threshold = 10.1  # above the achievable max (10.0) -- nothing should clear it
    db_session.commit()
    segments = _add_candidates(db_session, job, [(0.0, 6.0), (6.5, 9.0)])

    enqueued = []
    monkeypatch.setattr(
        "app.workers.scoring.enqueue", lambda stage, func, *a, **k: enqueued.append((stage, a, k))
    )

    scoring.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "scored"
    for s in segments:
        db_session.refresh(s)
        assert s.status == "rejected"
    assert enqueued == []  # nothing selected -- nothing to render


def test_scoring_selects_non_overlapping_over_naive_top_n(db_session, user, three_scene_video_path, monkeypatch):
    # Two heavily-overlapping high-signal windows plus one distinct
    # lower-signal window, max_clips=2 -- a naive top-2-by-score would
    # return the two overlapping windows (near-duplicates of the same
    # moment); non-max suppression should instead return one of those plus
    # the distinct one.
    job = _make_job(db_session, user, three_scene_video_path, max_clips=2)
    segments = _add_candidates(db_session, job, [(0.0, 6.0), (0.5, 5.5), (6.5, 9.0)])

    monkeypatch.setattr("app.workers.scoring.enqueue", lambda *a, **k: None)

    scoring.run(str(job.id))

    for s in segments:
        db_session.refresh(s)
    selected = [s for s in segments if s.status == "selected"]
    assert len(selected) == 2
    # The two overlapping high-signal windows (0,6) and (0.5,5.5) must not
    # both be selected.
    overlapping_ids = {segments[0].id, segments[1].id}
    assert not overlapping_ids.issubset({s.id for s in selected})
    assert segments[2].id in {s.id for s in selected}  # the distinct window


def test_scoring_no_transcript_fails_permanently(db_session, user):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="segmented")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    scoring.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "failed_scoring"
    assert "no transcript" in job.last_error


def test_scoring_no_candidates_fails_permanently(db_session, user, three_scene_video_path):
    job = _make_job(db_session, user, three_scene_video_path)

    scoring.run(str(job.id))

    db_session.refresh(job)
    assert job.status == "failed_scoring"
    assert "no pending_score candidate" in job.last_error


def test_scoring_degrades_gracefully_when_scene_detection_fails(db_session, user, three_scene_video_path, monkeypatch):
    job = _make_job(db_session, user, three_scene_video_path)
    _add_candidates(db_session, job, [(0.0, 6.0)])

    monkeypatch.setattr(
        "app.workers.scoring.detect_scene_cuts",
        lambda path, frame_skip=0: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("app.workers.scoring.enqueue", lambda *a, **k: None)

    scoring.run(str(job.id))

    db_session.refresh(job)
    # Scene detection failing shouldn't fail the whole job -- scoring should
    # still complete using the other signals.
    assert job.status == "scored"


def test_scoring_fallback_scene_detection_passes_configured_frame_skip(
    db_session, user, three_scene_video_path, monkeypatch
):
    job = _make_job(db_session, user, three_scene_video_path)
    _add_candidates(db_session, job, [(0.0, 6.0)])
    monkeypatch.setattr("app.workers.scoring.settings.scene_detect_frame_skip", 2)
    monkeypatch.setattr("app.workers.scoring.enqueue", lambda *a, **k: None)

    calls = []
    real = scoring.detect_scene_cuts

    def spy(video_path, frame_skip=0):
        calls.append(frame_skip)
        return real(video_path, frame_skip=frame_skip)

    monkeypatch.setattr("app.workers.scoring.detect_scene_cuts", spy)

    scoring.run(str(job.id))

    assert calls == [2]


def test_weights_from_settings_zeroes_audio_event_weights_when_disabled(monkeypatch):
    # Regression test for a real bug: SCORE_WEIGHT_LAUGHTER/
    # SCORE_WEIGHT_CROWD_REACTION default to 1.0 each even though
    # ENABLE_AUDIO_EVENT_SCORING defaults to False -- if those weights were
    # passed through unconditionally, score_out_of_10's denominator would
    # include 2.0 worth of weight no clip could ever earn while the flag is
    # off, silently deflating every score (see the fix's comment in
    # app.workers.scoring._weights_from_settings for the exact math: a
    # flawless clip on every real feature would cap at ~6.9/10 instead of
    # 10/10). Confirms configured nonzero weights get zeroed when disabled.
    monkeypatch.setattr("app.workers.scoring.settings.enable_audio_event_scoring", False)
    monkeypatch.setattr("app.workers.scoring.settings.score_weight_laughter", 1.0)
    monkeypatch.setattr("app.workers.scoring.settings.score_weight_crowd_reaction", 1.0)

    weights = scoring._weights_from_settings()

    assert weights.laughter == 0.0
    assert weights.crowd_reaction == 0.0


def test_weights_from_settings_uses_configured_weights_when_enabled(monkeypatch):
    monkeypatch.setattr("app.workers.scoring.settings.enable_audio_event_scoring", True)
    monkeypatch.setattr("app.workers.scoring.settings.score_weight_laughter", 2.0)
    monkeypatch.setattr("app.workers.scoring.settings.score_weight_crowd_reaction", 1.5)

    weights = scoring._weights_from_settings()

    assert weights.laughter == 2.0
    assert weights.crowd_reaction == 1.5


def test_weights_from_settings_always_passes_through_hook_strength(monkeypatch):
    # Unlike laughter/crowd_reaction, hook_strength isn't gated by an
    # enable_* flag -- it's a pure transcript-timing feature with no new
    # dependency, so it should come straight from settings regardless of
    # enable_audio_event_scoring's value.
    monkeypatch.setattr("app.workers.scoring.settings.enable_audio_event_scoring", False)
    monkeypatch.setattr("app.workers.scoring.settings.score_weight_hook_strength", 1.75)

    weights = scoring._weights_from_settings()

    assert weights.hook_strength == 1.75


def test_scoring_missing_job_is_a_noop(db_session):
    scoring.run(str(uuid.uuid4()))


def test_on_failure_marks_job_failed(db_session, user):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="segmented")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    fake_rq_job = type("FakeRqJob", (), {"args": [str(job.id)]})()
    scoring.on_failure(fake_rq_job, None, RuntimeError, RuntimeError("exhausted"), None)

    db_session.refresh(job)
    assert job.status == "failed_scoring"
    assert "exhausted" in job.last_error
