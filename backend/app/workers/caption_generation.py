"""Title/caption/hashtag-generation worker -- LLM-assisted enrichment
layered on top of scoring's deterministic default_caption placeholder
(app.core.scoring_logic.default_caption), NOT the burned-in on-screen
transcript subtitles (those are app.core.rendering_logic.build_srt,
applied inside app.workers.rendering).

NOT AUTO-INVOKED ANYMORE: this used to be enqueued by app.workers
.rendering.run right after a successful render, specifically so a
slow/failed LLM call could never block or fail the clip render itself.
Once the annotation gained a "title" field that gets burned into the
clip's own pixels (see app.core.rendering_logic.build_title_srt), that
ordering stopped working -- the title has to be known *before* rendering,
not after -- so app.workers.rendering.run now calls
app.core.caption_generation.generate_caption_annotation synchronously,
before building the ffmpeg filtergraph, and this worker is no longer
wired into anything automatically. It's kept as a manual regenerate path
(e.g. to redo a clip's title/hashtags/caption without re-rendering the
video) -- not yet exposed through an API endpoint, a natural next step if
that's ever needed (see README's "Next steps").

Trigger:  none automatic -- call app.workers.caption_generation.run(...)
          directly (or enqueue it by hand) to regenerate an existing
          clip's annotation.
Input:    candidate_segment_id
Output:   RenderedClip.caption_text overwritten with the generated caption
          (LLM's or the heuristic fallback's -- either way always a usable
          string); CandidateSegment.llm_annotation set to the full
          annotation dict (source/title/hashtags/caption/explanation/
          model/generated_at) -- surfaced to the API via RenderedClip's
          caption_title/caption_hashtags/caption_explanation/caption_source
          proxy properties (see app.db.models), same pattern as its
          score/score_breakdown proxies. Does NOT re-render the video, so
          a regenerated title never changes an already-burned-in banner.
State:    Does not touch stream_job/candidate_segment/rendered_clip status
          enums at all -- this is enrichment data, not a pipeline gate.
          Deliberately never marks anything 'failed': see
          app.core.caption_generation.generate_caption_annotation, which
          always returns a usable (if heuristic) annotation instead of
          raising, matching the project's reliability-first principle for a
          feature that's explicitly optional per the AI/ML rules ("LLM may
          assist... must never be the only thing choosing").
Retries:  0 -- nothing to retry; a transient LLM failure already degrades
          to the heuristic annotation inside generate_caption_annotation
          rather than raising, so an RQ-level retry would just re-run a
          call that already "succeeded" (with a fallback value). on_failure
          below only fires for a genuinely unexpected error (e.g. a DB
          hiccup between the two db_session blocks) and just logs it --
          the clip already has its existing annotation, so there is
          nothing to roll back.
"""
from __future__ import annotations

import uuid

from app.core.caption_generation import generate_caption_annotation
from app.db.models import CandidateSegment, RenderedClip, Transcript
from app.workers.common import db_session, logger


def run(candidate_segment_id: str) -> None:
    log = logger.bind(candidate_segment_id=candidate_segment_id, worker="caption_generation")
    log.info("caption_generation.start")

    with db_session() as db:
        candidate = db.get(CandidateSegment, uuid.UUID(candidate_segment_id))
        if candidate is None:
            log.warning("caption_generation.candidate_not_found")
            return

        clip = (
            db.query(RenderedClip)
            .filter(RenderedClip.candidate_segment_id == candidate.id)
            .order_by(RenderedClip.created_at.desc())
            .first()
        )
        if clip is None:
            log.warning("caption_generation.clip_not_found")
            return

        transcript = db.query(Transcript).filter_by(stream_job_id=candidate.stream_job_id).one_or_none()
        transcript_segments = list(transcript.segments) if transcript else []
        start = float(candidate.start_seconds)
        end = float(candidate.end_seconds)
        score_breakdown = candidate.score_breakdown
        existing_caption = clip.caption_text or ""
        clip_id = clip.id
        candidate_id = candidate.id

    annotation = generate_caption_annotation(transcript_segments, start, end, score_breakdown, existing_caption)

    with db_session() as db:
        candidate = db.get(CandidateSegment, candidate_id)
        clip = db.get(RenderedClip, clip_id)
        if candidate is None or clip is None:
            # Rows were deleted (e.g. a reviewer deleted the clip) between
            # the two db_session blocks above -- nothing left to annotate.
            log.warning("caption_generation.rows_gone_before_write")
            return
        candidate.llm_annotation = annotation
        clip.caption_text = annotation["caption"]

    log.info("caption_generation.done", source=annotation["source"], hashtag_count=len(annotation["hashtags"]))


def on_failure(job, connection, type, value, traceback) -> None:
    """RQ failure callback -- only reached for a genuinely unexpected error
    (see module docstring); nothing in the DB needs to change since
    generate_caption_annotation itself never raises."""
    candidate_segment_id = job.args[0] if job.args else None
    logger.bind(candidate_segment_id=candidate_segment_id, worker="caption_generation").error(
        "caption_generation.unexpected_failure", error=str(value)
    )
