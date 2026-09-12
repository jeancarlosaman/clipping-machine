"""Scoring worker -- architecture doc §7 / §9.

Trigger:  stream_job.segmented (enqueued by app.workers.segmentation.run)
Input:    stream_job_id, its transcript, and its pending_score
          candidate_segments
Output:   every candidate_segment moved to status='selected' (up to
          job.max_clips, by score_out_of_10, via non-max suppression so
          overlapping candidates of the same moment don't crowd out
          distinct ones -- see app.core.scoring_logic.select_top_non_overlapping;
          also filtered by job.min_score_threshold if set) or 'rejected',
          with features/score/score_breakdown written; one `rendered_clips`
          row (status='pending') created per selected candidate, and one
          rendering job enqueued per clip
State:    segmented -> scoring -> scored | failed_scoring
Retries:  2x -- deterministic CPU work (no external calls), same reasoning
          as segmentation.py: retries only help with transient I/O (a flaky
          storage read while re-deriving scene cuts), not logic errors.

Ranking is deterministic feature scoring only (app.core.scoring_logic) --
per the project's AI/ML principles, no LLM chooses or reorders clips here.
LLM/agent reasoning may later *assist* (e.g. rewriting default_caption's
placeholder, or explaining a clip's score to the reviewer) but that's a
post-MVP addition layered on top of this, not a replacement for it.
"""
from __future__ import annotations

import os
import tempfile
import uuid

from app.core.config import settings
from app.core.queue import enqueue
from app.core.rendering_logic import normalize_parts, parts_duration
from app.core.scoring_logic import (
    ScoreWeights,
    default_caption,
    score_multipart_window,
    score_window,
    select_top_non_overlapping,
)
from app.core.segmentation_logic import detect_scene_cuts
from app.core.storage import storage
from app.db.models import CandidateSegment, RenderedClip, StreamJob, Transcript
from app.workers import rendering
from app.workers.common import audio_object_key, db_session, estimate_job_timeout_seconds, job_is_cancelled, logger


def _weights_from_settings() -> ScoreWeights:
    # BUGFIX (see README's "Scoring" section, "a real regression found and
    # fixed"): laughter/crowd_reaction raw values are ALWAYS 0.0 when
    # ENABLE_AUDIO_EVENT_SCORING is off (score_window never receives an
    # audio_events dict in that case -- see _compute_audio_events below,
    # which returns {} immediately when the flag is off). If their weights
    # were passed through unconditionally, score_out_of_10's denominator
    # (sum of ALL configured weights, see scoring_logic.score_out_of_10)
    # would include 2.0 worth of weight that NO clip could ever earn while
    # the flag is off -- silently deflating every single score (a "perfect"
    # clip on every real feature capped at ~6.9/10 instead of 10/10 with the
    # stock SCORE_WEIGHT_* defaults). Zeroing these out when the flag is off
    # keeps score_out_of_10 identical to how it behaved before this feature
    # existed, exactly like motion_proxy's weight already defaults to 0.0
    # in config.py specifically because *that* feature isn't implemented --
    # same principle, applied here for "implemented but not enabled."
    audio_events_active = settings.enable_audio_event_scoring
    return ScoreWeights(
        speech_density=settings.score_weight_speech_density,
        pause_count=settings.score_weight_pause_count,
        question_marks=settings.score_weight_question_marks,
        emotional_language=settings.score_weight_emotional_language,
        scene_changes=settings.score_weight_scene_changes,
        motion_proxy=settings.score_weight_motion_proxy,
        laughter=settings.score_weight_laughter if audio_events_active else 0.0,
        crowd_reaction=settings.score_weight_crowd_reaction if audio_events_active else 0.0,
        # Unlike laughter/crowd_reaction, not gated by a feature flag --
        # hook_strength is computed straight from transcript segments every
        # caller already has, same as speech_density/question_marks. See
        # scoring_logic.py's module docstring and config.py's
        # score_weight_hook_strength comment.
        hook_strength=settings.score_weight_hook_strength,
    )


def _compute_audio_events(
    stream_job_id: str, candidate_rows: list[dict], log
) -> dict[uuid.UUID, dict]:
    """Runs PANNs audio-event detection once per candidate window, reusing
    one loaded model across all of them (see
    app.core.audio_events.AudioEventDetector's docstring for why loading it
    per-window would be wasteful). Never raises -- a failure here (model
    load, missing audio file, etc.) degrades every window's audio_events to
    "not computed" (score_window then treats laughter/crowd_reaction as
    0.0, same as if the flag were off) rather than failing the whole
    scoring job over an optional signal, same posture as this file's
    existing scene-cut-detection-failure handling above.
    """
    if not settings.enable_audio_event_scoring:
        return {}
    try:
        from app.core.audio_events import get_audio_event_detector

        detector = get_audio_event_detector()
    except Exception as exc:
        log.warning("scoring.audio_event_detector_unavailable", error=str(exc))
        return {}

    tmp_dir = tempfile.mkdtemp(prefix=f"score-audio-{stream_job_id}-")
    try:
        local_audio_path = storage.get_local_path(
            audio_object_key(stream_job_id), download_to=os.path.join(tmp_dir, "audio.wav")
        )
    except Exception as exc:
        log.warning("scoring.audio_events_degraded", error=str(exc))
        return {}

    events_by_id: dict[uuid.UUID, dict] = {}
    try:
        for row in candidate_rows:
            events_by_id[row["id"]] = detector.score_window(local_audio_path, row["start"], row["end"])
    finally:
        if os.path.exists(local_audio_path):
            os.remove(local_audio_path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)
    return events_by_id


def run(stream_job_id: str) -> None:
    log = logger.bind(stream_job_id=stream_job_id, worker="scoring")
    log.info("scoring.start")

    # Cooperative cancel (see app.workers.common.job_is_cancelled): stop at
    # this stage boundary instead of doing the work and enqueuing the next
    # stage. Returning rather than raising keeps this out of the failure
    # path -- a cancelled job is not a failed one, and must not burn retries
    # or trip the on_failure callback.
    if job_is_cancelled(stream_job_id):
        log.info("scoring.cancelled")
        return

    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job is None:
            # Not retryable -- the row doesn't exist, retrying changes nothing.
            log.error("scoring.job_not_found")
            return
        job.status = "scoring"
        job.retry_count = job.retry_count + 1
        raw_object_key = job.raw_object_key
        max_clips = job.max_clips
        min_score_threshold = (
            float(job.min_score_threshold)
            if job.min_score_threshold is not None
            else settings.default_min_score_threshold
        )
        # Persisted by segmentation.py -- reuse instead of re-detecting when
        # present (see below); None means segmentation hasn't run/persisted
        # yet (older row, or segmentation itself failed before getting
        # there), an empty list means detection ran and genuinely found no
        # cuts. Both are meaningfully different from "not computed".
        cached_scene_cuts = job.scene_cuts_seconds

        transcript = db.query(Transcript).filter_by(stream_job_id=job.id).one_or_none()
        transcript_segments = list(transcript.segments) if transcript else []

        candidates = (
            db.query(CandidateSegment)
            .filter_by(stream_job_id=job.id, status="pending_score")
            .order_by(CandidateSegment.start_seconds)
            .all()
        )
        # Pull out plain values now -- these ORM objects are bound to this
        # session, which closes at the end of the `with` block.
        candidate_rows = [
            {
                "id": c.id,
                "start": float(c.start_seconds),
                "end": float(c.end_seconds),
                # None for an ordinary contiguous candidate; a validated
                # list of (start, end) pairs for a stitched multi-part one
                # (see CandidateSegment.parts).
                "parts": normalize_parts(c.parts),
            }
            for c in candidates
        ]

    if not transcript_segments:
        _mark_failed(stream_job_id, "no transcript segments available to score against")
        log.error("scoring.no_transcript")
        return

    if not candidate_rows:
        _mark_failed(stream_job_id, "no pending_score candidate segments to score")
        log.error("scoring.no_candidates")
        return

    if cached_scene_cuts is not None:
        # Common path: segmentation.py already computed and persisted these.
        # Skips a second full-video download + PySceneDetect pass entirely
        # (both real cost: decode time, and for S3-backed storage, a second
        # network fetch of the whole raw video) -- see StreamJob.scene_cuts_seconds.
        scene_cuts = cached_scene_cuts
        log.info("scoring.scene_cuts_reused", scene_cut_count=len(scene_cuts))
    else:
        # Fallback for a row from before this cache existed, or where
        # segmentation didn't get far enough to persist it. Re-derive the
        # old way. Unlike segmentation.py, a failure here is NOT fatal:
        # scene_changes is one signal of several, and scoring can still
        # produce a meaningful ranking without it (its weighted contribution
        # just drops to 0 for every candidate) -- failing the whole job over
        # a visual-cue signal would be a worse trade-off than degrading
        # gracefully.
        scene_cuts = []
        tmp_dir = tempfile.mkdtemp(prefix=f"score-{stream_job_id}-")
        try:
            try:
                local_video_path = storage.get_local_path(
                    raw_object_key, download_to=os.path.join(tmp_dir, "source")
                )
                scene_cuts = detect_scene_cuts(local_video_path, frame_skip=settings.scene_detect_frame_skip)
            except Exception as exc:
                log.warning("scoring.scene_detection_degraded", error=str(exc))
                scene_cuts = []
        finally:
            source_path = os.path.join(tmp_dir, "source")
            if os.path.exists(source_path):
                os.remove(source_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)

    audio_events_by_id = _compute_audio_events(stream_job_id, candidate_rows, log)

    weights = _weights_from_settings()
    scored = []
    for row in candidate_rows:
        # A stitched candidate is scored over the union of its parts, not
        # the span they're drawn from -- see score_multipart_window for why
        # span-based features would be actively wrong here.
        if row["parts"]:
            composite, breakdown = score_multipart_window(
                transcript_segments, row["parts"], scene_cuts, weights,
                audio_events=audio_events_by_id.get(row["id"]),
            )
            caption = " ".join(
                default_caption(transcript_segments, p_start, p_end) for p_start, p_end in row["parts"]
            ).strip()
        else:
            composite, breakdown = score_window(
                transcript_segments, row["start"], row["end"], scene_cuts, weights,
                audio_events=audio_events_by_id.get(row["id"]),
            )
            caption = default_caption(transcript_segments, row["start"], row["end"])
        scored.append(
            {
                **row,
                "composite": composite,
                "breakdown": breakdown,
                "caption": caption,
                "score_10": breakdown["score_out_of_10"],
            }
        )

    # Non-max suppression instead of a naive top-N-by-score slice: candidate
    # generation (app.core.segmentation_logic's sliding sub-windows for long
    # speech blocks) can produce heavily overlapping candidates covering the
    # same moment, and taking the raw top N by score could return several
    # near-duplicates of one highlight instead of N distinct ones. See
    # app.core.scoring_logic.select_top_non_overlapping's docstring.
    # min_score_threshold (per-job override, default 0 = no filtering) means
    # fewer than max_clips may end up selected if that's genuinely all that
    # clears the bar -- that's the intended behavior of a "minimum score to
    # look for" setting, not a bug.
    selected_ids = select_top_non_overlapping(
        [{"id": r["id"], "start": r["start"], "end": r["end"], "score": r["score_10"]} for r in scored],
        max_count=max_clips,
        min_score=min_score_threshold,
    )
    scored.sort(key=lambda r: r["score_10"], reverse=True)

    rendering_enqueue_ids: list[tuple[uuid.UUID, float]] = []
    with db_session() as db:
        for row in scored:
            candidate = db.get(CandidateSegment, row["id"])
            candidate.features = row["breakdown"]["raw"]
            # 0..10, not the raw weighted composite -- see
            # app.core.scoring_logic.score_out_of_10. The raw composite is
            # still available in score_breakdown["composite"] for anyone
            # debugging the weighting itself.
            candidate.score = row["score_10"]
            candidate.score_breakdown = row["breakdown"]
            if row["id"] in selected_ids:
                candidate.status = "selected"
                # Real playing time: the sum of a stitched clip's parts,
                # or the plain window length for an ordinary one. Used both
                # for the stored duration a reviewer sees and for the
                # render job's timeout budget below.
                clip_duration = parts_duration(row["parts"]) if row["parts"] else row["end"] - row["start"]
                clip = RenderedClip(
                    candidate_segment_id=candidate.id,
                    stream_job_id=uuid.UUID(stream_job_id),
                    status="pending",
                    caption_text=row["caption"],
                    duration_seconds=clip_duration,
                )
                db.add(clip)
                rendering_enqueue_ids.append((candidate.id, clip_duration))
            else:
                candidate.status = "rejected"

        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        job.status = "scored"

    log.info(
        "scoring.done",
        candidate_count=len(scored),
        selected_count=len(selected_ids),
        scene_cut_count=len(scene_cuts),
    )

    # 3x retries per clip, isolated -- see rendering.py's docstring. One
    # clip's ffmpeg failure must not affect its siblings, which is why
    # rendering is enqueued per-candidate rather than once for the job.
    #
    # Timeout is based on the *clip's* own duration, not the whole video --
    # candidate windows are bounded by SEGMENT_MAX_CLIP_SECONDS (default
    # 90s), so even a long VOD's rendering jobs stay short. multiplier=6 is
    # generous for a slow CPU doing libx264 encode + libass caption burn-in
    # on a <=90s clip.
    for candidate_id, clip_duration in rendering_enqueue_ids:
        render_timeout = estimate_job_timeout_seconds(clip_duration, multiplier=6.0, minimum=180)
        enqueue(
            "rendering",
            rendering.run,
            str(candidate_id),
            max_retries=3,
            on_failure=rendering.on_failure,
            job_timeout=render_timeout,
        )


def _mark_failed(stream_job_id: str, error: str) -> None:
    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job:
            job.status = "failed_scoring"
            job.last_error = error[:4000]


def on_failure(job, connection, type, value, traceback) -> None:
    """RQ failure callback -- fires once retries are exhausted (see app.core.queue.enqueue)."""
    stream_job_id = job.args[0]
    _mark_failed(stream_job_id, f"{type.__name__}: {value}")
    logger.bind(stream_job_id=stream_job_id, worker="scoring").error(
        "scoring.retries_exhausted", error=str(value)
    )
