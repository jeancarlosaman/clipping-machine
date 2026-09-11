"""Segmentation worker -- architecture doc §7.

Trigger:  stream_job.transcribed (enqueued by app.workers.transcription.run)
Input:    stream_job_id, the job's transcript, the raw video
Output:   `candidate_segments` rows (start/end windows, status=pending_score)
State:    transcribed -> segmenting -> segmented | failed_segmentation
Retries:  2x -- this is deterministic CPU work (scene detection + a pure
          window-building function), so retries only cover transient I/O
          (a flaky storage read), not logic errors that would just fail the
          same way twice. A missing transcript, an unreadable video, or a
          transcript that produces zero usable windows are all treated as
          permanent -- same "fail clearly once, don't burn retries on
          something that won't change" pattern as ingest.py.

Combines PySceneDetect's visual scene-cut detection with transcript-derived
speech/silence structure to propose candidate clip windows -- the actual
window-building logic lives in app/core/segmentation_logic.py, kept
separate and pure (no DB/IO) so it's unit-testable without a real video.
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.llm_segmentation import generate_llm_segment_suggestions
from app.core.queue import enqueue
from app.core.scoring_logic import window_text
from app.core.segmentation_logic import build_candidate_windows, detect_scene_cuts
from app.core.storage import storage
from app.db.models import CandidateSegment, RenderedClip, ReviewDecision, StreamJob, Transcript
from app.workers import scoring
from app.workers.common import db_session, estimate_job_timeout_seconds, logger


def run(stream_job_id: str) -> None:
    log = logger.bind(stream_job_id=stream_job_id, worker="segmentation")
    log.info("segmentation.start")

    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job is None:
            # Not retryable -- the row doesn't exist, retrying changes nothing.
            log.error("segmentation.job_not_found")
            return
        job.status = "segmenting"
        job.retry_count = job.retry_count + 1
        raw_object_key = job.raw_object_key
        duration_seconds = job.duration_seconds  # captured now -- job is detached once this session closes
        user_id = job.user_id  # for the few-shot feedback examples below (this creator's own past reviews)
        # Per-job overrides (see StreamJob.min_clip_seconds/max_clip_seconds) --
        # None means "use the global settings.* default", same as before
        # these existed.
        min_len = float(job.min_clip_seconds) if job.min_clip_seconds is not None else settings.segment_min_clip_seconds
        max_len = float(job.max_clip_seconds) if job.max_clip_seconds is not None else settings.segment_max_clip_seconds

        transcript = db.query(Transcript).filter_by(stream_job_id=job.id).one_or_none()
        transcript_segments = list(transcript.segments) if transcript else []

    if min_len >= max_len:
        # Defense in depth -- app.api.routers.stream_jobs validates this at
        # job-creation time (comparing each side's EFFECTIVE value, not just
        # what a request explicitly submitted), but this worker has no
        # guarantee every candidate_segments row ever came through that
        # endpoint (an old job created before that validation existed, a
        # manual DB edit, a future caller). Real bug this caught in
        # production: a per-job min_clip_seconds=35 override combined with
        # a stale SEGMENT_MAX_CLIP_SECONDS=10 in .env produced min_len=35 >
        # max_len=10 -- build_candidate_windows/_sliding_subwindows has no
        # guard against that, so `step = max(max_len/3, min_len)` collapsed
        # to min_len and every "candidate" came out ~max_len long (~10s)
        # spaced ~min_len apart (~35s), silently violating the very minimum
        # the user had just set. Failing clearly here beats generating 60+
        # garbage candidates that then get scored and rendered.
        _mark_failed(
            stream_job_id,
            f"min_clip_seconds ({min_len}) must be less than max_clip_seconds ({max_len}) -- "
            "check this job's overrides and SEGMENT_MIN_CLIP_SECONDS/SEGMENT_MAX_CLIP_SECONDS in .env",
        )
        log.error("segmentation.invalid_clip_length_range", min_len=min_len, max_len=max_len)
        return

    if not transcript_segments:
        _mark_failed(stream_job_id, "no transcript segments available to segment from")
        log.error("segmentation.no_transcript")
        return

    tmp_dir = tempfile.mkdtemp(prefix=f"segment-{stream_job_id}-")
    phase_seconds: dict[str, float] = {}
    started_at = time.perf_counter()
    try:
        phase_start = time.perf_counter()
        local_video_path = storage.get_local_path(
            raw_object_key, download_to=os.path.join(tmp_dir, "source")
        )
        phase_seconds["download"] = time.perf_counter() - phase_start

        # The LLM proposer and scene detection are completely independent --
        # one reads the transcript and talks to Ollama/OpenAI over a socket,
        # the other decodes video frames locally. Run CONCURRENTLY rather
        # than one after the other, so this stage costs max(llm, scene)
        # instead of llm + scene. The LLM call spends essentially its whole
        # life blocked on a socket (minutes, for an 8B model on CPU), so it
        # releases the GIL and genuinely overlaps with the decode loop
        # instead of competing with it.
        #
        # A single worker thread, not a pool: there is exactly one job to
        # overlap. Each thread builds its own Session via db_session()
        # (SessionLocal is a plain sessionmaker, so that is a fresh Session
        # per call, not a shared one) -- see _gather_feedback_examples.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-segment")
        try:
            llm_started = time.perf_counter()
            llm_future = executor.submit(
                _llm_suggestions_safely, transcript_segments, min_len, max_len, user_id, log
            )

            phase_start = time.perf_counter()
            try:
                scene_cuts = detect_scene_cuts(local_video_path, frame_skip=settings.scene_detect_frame_skip)
            except Exception as exc:
                # PySceneDetect/OpenCV raising here overwhelmingly means an
                # unreadable/unsupported video, not something a retry fixes --
                # ingest already validated the file once, but codecs OpenCV's
                # backend can't handle are a different failure mode than ffprobe's.
                #
                # The in-flight LLM call is left to finish on its own rather
                # than cancelled (a Future already running can't be), so a
                # job that fails here still burns one LLM call. Acceptable:
                # it costs nothing but a little Ollama time on a job that
                # was going to fail regardless.
                _mark_failed(stream_job_id, f"scene detection failed: {exc}")
                log.error("segmentation.permanent_failure", error=str(exc))
                return
            phase_seconds["scene_detect"] = time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            windows = build_candidate_windows(
                transcript_segments,
                scene_cuts,
                min_len=min_len,
                max_len=max_len,
                max_gap_seconds=settings.segment_silence_gap_seconds,
            )
            phase_seconds["build_windows"] = time.perf_counter() - phase_start

            # Whatever is left of the LLM call after scene detection already
            # ran. "llm_wall" is the call's own total duration; the gap
            # between it and the stage total is what the overlap actually
            # saved.
            llm_suggestions = llm_future.result()
            phase_seconds["llm_wall"] = time.perf_counter() - llm_started
        finally:
            executor.shutdown(wait=False)

        if not windows and not llm_suggestions:
            _mark_failed(
                stream_job_id,
                f"no candidate windows found -- transcript had no speech block at least {min_len}s long",
            )
            log.warning("segmentation.no_windows", scene_cut_count=len(scene_cuts))
            return

        with db_session() as db:
            for start, end in windows:
                db.add(
                    CandidateSegment(
                        stream_job_id=uuid.UUID(stream_job_id),
                        start_seconds=start,
                        end_seconds=end,
                        status="pending_score",
                        origin="heuristic",
                    )
                )
            for suggestion in llm_suggestions:
                db.add(
                    CandidateSegment(
                        stream_job_id=uuid.UUID(stream_job_id),
                        start_seconds=suggestion["start"],
                        end_seconds=suggestion["end"],
                        status="pending_score",
                        origin="llm",
                        llm_reason=suggestion["reason"],
                        # Only present for a stitched multi-part moment
                        # (see CandidateSegment.parts); a normal single-
                        # range suggestion stores null here and behaves
                        # exactly like a heuristic candidate downstream.
                        parts=suggestion.get("parts"),
                    )
                )
            job = db.get(StreamJob, uuid.UUID(stream_job_id))
            job.status = "segmented"
            # Persist so scoring.py can reuse these instead of re-downloading
            # the raw video and re-running PySceneDetect from scratch -- see
            # app/workers/scoring.py's docstring. Persisted even when empty
            # (scene_cuts == []) so scoring can tell "no cuts found" apart
            # from "not computed yet" via the column being non-null.
            job.scene_cuts_seconds = scene_cuts

        total_seconds = time.perf_counter() - started_at
        # Emitted as its own event so it is greppable and so the numbers are
        # available WITHOUT re-running anything -- "segmentation feels slow"
        # was previously unanswerable without guessing which half was to
        # blame. `llm_wall` overlapping `scene_detect` is the point: if the
        # two roughly match the total, the overlap is working.
        log.info(
            "segmentation.phase_timing",
            total_seconds=round(total_seconds, 1),
            overlap_saved_seconds=round(
                max(0.0, sum(phase_seconds.get(k, 0.0) for k in ("scene_detect", "llm_wall"))
                    - max(phase_seconds.get("scene_detect", 0.0), phase_seconds.get("llm_wall", 0.0))),
                1,
            ),
            **{k: round(v, 1) for k, v in phase_seconds.items()},
        )
        log.info(
            "segmentation.done",
            window_count=len(windows),
            llm_suggestion_count=len(llm_suggestions),
            scene_cut_count=len(scene_cuts),
        )

        # multiplier=3: generous ceiling for scoring's *fallback* path (this
        # job's own scene_cuts_seconds, just persisted above, means scoring
        # normally reuses it and skips scene detection entirely -- see
        # scoring.py). Kept at the old, safe value rather than lowered,
        # since the fallback path (an older row, or a case where this
        # persist somehow didn't happen) still does the full re-download +
        # re-detect and needs the same margin segmentation's own timeout
        # below does. A too-generous timeout costs nothing when unused; a
        # too-tight one risks a false failure on the rare path that needs it.
        scoring_timeout = estimate_job_timeout_seconds(duration_seconds, multiplier=3.0, minimum=300)

        # max_retries=2 to match scoring.py's documented retry policy for
        # its deterministic pass.
        enqueue(
            "scoring", scoring.run, stream_job_id, max_retries=2,
            on_failure=scoring.on_failure, job_timeout=scoring_timeout,
        )

    finally:
        source_path = os.path.join(tmp_dir, "source")
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


def _llm_suggestions_safely(transcript_segments, min_len, max_len, user_id, log) -> list[dict]:
    """The LLM proposer plus its few-shot example lookup, as one unit that
    never raises -- this runs on a worker thread (see run()), where an
    escaping exception would surface only when the Future is resolved and
    would otherwise take the whole segmentation job down with it.

    generate_llm_segment_suggestions is itself documented never to raise;
    this is defense in depth for that promise plus the DB work in
    _gather_feedback_examples, exactly as the inline try/except did before
    this moved onto a thread.
    """
    try:
        feedback_examples = _gather_feedback_examples(user_id, log)
        return generate_llm_segment_suggestions(
            transcript_segments, min_len, max_len, feedback_examples=feedback_examples
        )
    except Exception as exc:
        log.warning("segmentation.llm_suggestions_failed", error=str(exc))
        return []


def _gather_feedback_examples(user_id, log) -> list[dict]:
    """This creator's most recent reviewed clips that carry a written
    comment, as `{"decision", "notes", "transcript"}` dicts for the LLM
    segment proposer's few-shot block (see
    app.core.llm_segmentation_logic.build_feedback_examples_block).

    Only reviews with an actual comment count: an approve/reject with no
    written reasoning says nothing about *why*, and the deterministic
    scorer already covers "was it picked." Both approvals and rejections
    are included on purpose -- knowing what this creator throws away is at
    least as informative as knowing what they keep, and negatives are the
    thing a purely "here are good clips" example set can never teach.

    The transcript excerpt comes from the clip's own candidate window
    against its job's transcript (the same `window_text` scoring uses),
    NOT from the stored caption -- a caption may have been LLM-rewritten
    or hand-edited, so it isn't a faithful record of what was actually
    said in the clip the reviewer was judging.

    Never raises: any failure here degrades to no examples (the prompt
    then reads exactly as it did before this feature existed) rather than
    disturbing segmentation, same posture as the LLM call it feeds.
    """
    limit = settings.llm_segment_feedback_examples
    if not settings.enable_llm_segment_suggestions or limit <= 0:
        return []

    try:
        with db_session() as db:
            rows = (
                db.query(ReviewDecision, CandidateSegment)
                .join(RenderedClip, ReviewDecision.rendered_clip_id == RenderedClip.id)
                .join(CandidateSegment, RenderedClip.candidate_segment_id == CandidateSegment.id)
                .filter(ReviewDecision.user_id == user_id)
                .filter(ReviewDecision.notes.isnot(None))
                .filter(ReviewDecision.notes != "")
                .order_by(ReviewDecision.decided_at.desc())
                .limit(limit)
                .all()
            )

            # One transcript fetch per distinct source job, not per example
            # -- several reviewed clips usually come from the same VOD.
            transcripts: dict[uuid.UUID, list[dict]] = {}
            examples: list[dict] = []
            for decision, candidate in rows:
                job_id = candidate.stream_job_id
                if job_id not in transcripts:
                    transcript = db.query(Transcript).filter_by(stream_job_id=job_id).one_or_none()
                    transcripts[job_id] = list(transcript.segments) if transcript else []
                excerpt = window_text(
                    transcripts[job_id], float(candidate.start_seconds), float(candidate.end_seconds)
                )
                examples.append(
                    {"decision": decision.decision, "notes": decision.notes, "transcript": excerpt}
                )
    except Exception as exc:
        log.warning("segmentation.feedback_examples_failed", error=str(exc))
        return []

    if examples:
        log.info("segmentation.feedback_examples_loaded", count=len(examples))
    return examples


def _mark_failed(stream_job_id: str, error: str) -> None:
    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job:
            job.status = "failed_segmentation"
            job.last_error = error[:4000]


def on_failure(job, connection, type, value, traceback) -> None:
    """RQ failure callback -- fires once retries are exhausted (see app.core.queue.enqueue)."""
    stream_job_id = job.args[0]
    _mark_failed(stream_job_id, f"{type.__name__}: {value}")
    logger.bind(stream_job_id=stream_job_id, worker="segmentation").error(
        "segmentation.retries_exhausted", error=str(value)
    )
