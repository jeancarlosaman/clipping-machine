"""POST /api/v1/clips/{clip_id}/review, /upload -- architecture doc §6.

Also serves the rendered video/thumbnail bytes for the dev console's
preview player (GET /video, /thumbnail) -- see the note on those routes for
why this is fine for the MVP dev console but not how production serving
would work.
"""
from __future__ import annotations

import os
import tempfile
import uuid

from fastapi import APIRouter
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUserDep, DbDep
from app.api.errors import ApiError
from app.core.caption_logic import MAX_CAPTION_CHARS, MAX_HASHTAGS, MAX_TITLE_CHARS, normalize_hashtags
from app.core.queue import enqueue
from app.core.storage import storage
from app.db.models import CreatorAccount, RenderedClip, ReviewDecision, UploadTask, User
from app.schemas import ClipCaptionUpdate, RenderedClipOut, ReviewDecisionOut, ReviewRequest, UploadRequest, UploadTaskOut
from app.workers import upload as upload_worker

router = APIRouter(prefix="/api/v1/clips", tags=["clips"])


@router.post("/{clip_id}/review", response_model=ReviewDecisionOut, status_code=201)
def review_clip(
    clip_id: uuid.UUID, body: ReviewRequest, db: Session = DbDep, user: User = CurrentUserDep
) -> ReviewDecision:
    clip = _get_owned_clip(db, user, clip_id)
    if clip.status != "rendered":
        raise ApiError(
            409, "clip_not_ready",
            f"Clip is not ready for review (status={clip.status})",
        )

    decision = ReviewDecision(
        rendered_clip_id=clip.id, user_id=user.id, decision=body.decision, notes=body.notes, rating=body.rating
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)
    return decision


@router.patch("/{clip_id}/caption", response_model=RenderedClipOut)
def update_clip_caption(
    clip_id: uuid.UUID, body: ClipCaptionUpdate, db: Session = DbDep, user: User = CurrentUserDep
) -> RenderedClip:
    """Manual edit of a clip's generated title/hashtags/caption (see
    app.core.caption_generation for how they're first produced, either by
    the LLM or its heuristic fallback, synchronously during rendering) --
    lets a reviewer correct or personalize them before upload. Per the
    project's "human review required by default" principle, that review
    probably ought to cover this text too, not just clip approve/reject
    (see README's "Next steps"). Stored the same place the generated
    version was (CandidateSegment.llm_annotation / RenderedClip
    .caption_text -- no new columns), with `source` flipped to
    `'manual_edit'` so the dev console's `✨ AI`/`heuristic` badge reflects
    the change instead of still claiming a source that's no longer
    accurate. NOTE: editing `title` here does NOT re-render the video --
    the on-screen banner already burned into the clip's pixels (if
    ENABLE_CLIP_TITLE_OVERLAY was on at render time) stays whatever it was;
    this only updates the stored record. See RenderedClip.caption_title's
    docstring.
    """
    clip = _get_owned_clip(db, user, clip_id)
    if body.hashtags is None and body.caption is None and body.title is None:
        raise ApiError(400, "no_fields_to_update", "Provide at least one of hashtags/caption/title")

    candidate = clip.candidate_segment
    annotation = dict(candidate.llm_annotation or {})

    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise ApiError(400, "invalid_title", "title must not be empty")
        annotation["title"] = title[:MAX_TITLE_CHARS]

    if body.hashtags is not None:
        hashtags = normalize_hashtags(body.hashtags, max_hashtags=MAX_HASHTAGS)
        if not hashtags:
            raise ApiError(400, "invalid_hashtags", "hashtags must contain at least one usable entry")
        annotation["hashtags"] = hashtags

    if body.caption is not None:
        caption = body.caption.strip()
        if not caption:
            raise ApiError(400, "invalid_caption", "caption must not be empty")
        annotation["caption"] = caption[:MAX_CAPTION_CHARS]
        clip.caption_text = annotation["caption"]

    annotation["source"] = "manual_edit"
    annotation.setdefault("title", "")
    annotation.setdefault("explanation", "")
    annotation.setdefault("model", None)
    # Reassign (not mutate in place) so SQLAlchemy's normal attribute-set
    # change tracking picks it up -- same pattern as scoring.py's
    # candidate.score_breakdown = ... assignment; JSON/JSONB columns don't
    # auto-detect in-place dict mutation.
    candidate.llm_annotation = annotation

    db.commit()
    db.refresh(clip)
    return clip


@router.post("/{clip_id}/upload", response_model=UploadTaskOut, status_code=202)
def upload_clip(
    clip_id: uuid.UUID, body: UploadRequest, db: Session = DbDep, user: User = CurrentUserDep
) -> UploadTask:
    clip = _get_owned_clip(db, user, clip_id)

    latest_decision = db.scalars(
        select(ReviewDecision)
        .where(ReviewDecision.rendered_clip_id == clip.id)
        .order_by(ReviewDecision.decided_at.desc())
        .limit(1)
    ).first()
    if latest_decision is None or latest_decision.decision != "approved":
        raise ApiError(409, "review_required", "Clip must have an approved review decision before upload")

    account = db.get(CreatorAccount, body.creator_account_id)
    if account is None or account.user_id != user.id:
        raise ApiError(404, "not_found", "Creator account not found")

    task = UploadTask(
        rendered_clip_id=clip.id,
        creator_account_id=account.id,
        platform=body.platform,
        target_mode=body.target_mode,
        status="queued",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    enqueue("upload", upload_worker.run, str(task.id), on_failure=upload_worker.on_failure)

    return task


@router.delete("/{clip_id}", status_code=204, response_model=None)
def delete_clip(clip_id: uuid.UUID, db: Session = DbDep, user: User = CurrentUserDep) -> None:
    """Deletes a single rendered clip -- e.g. one you've reviewed and
    rejected and don't want cluttering the review queue anymore.
    review_decisions/upload_tasks under it cascade at the DB level
    (ondelete="CASCADE", see app.db.models). Allowed at any clip status,
    not just 'rejected' -- there's no real harm in deleting a pending or
    even an approved-but-not-yet-uploaded clip if that's what's wanted.
    """
    clip = _get_owned_clip(db, user, clip_id)
    if clip.object_key:
        storage.delete(clip.object_key)
    if clip.thumbnail_key:
        storage.delete(clip.thumbnail_key)
    db.delete(clip)
    db.commit()


@router.get("/{clip_id}/video")
def get_clip_video(clip_id: uuid.UUID, db: Session = DbDep, user: User = CurrentUserDep) -> FileResponse:
    """Streams the rendered clip's mp4 bytes -- built for the dev console's
    preview player, not production serving. For S3-backed storage this
    downloads the whole object to a temp file per request; a real
    creator-facing product would use signed URLs / a CDN in front of
    storage instead of proxying bytes through the API process.
    """
    clip = _get_owned_clip(db, user, clip_id)
    if not clip.object_key:
        raise ApiError(404, "not_found", "Clip has no rendered video yet")
    local_path = storage.get_local_path(
        clip.object_key, download_to=os.path.join(tempfile.gettempdir(), f"clip-video-{clip.id}.mp4")
    )
    return FileResponse(local_path, media_type="video/mp4", filename=f"{clip.id}.mp4")


@router.get("/{clip_id}/thumbnail")
def get_clip_thumbnail(clip_id: uuid.UUID, db: Session = DbDep, user: User = CurrentUserDep) -> FileResponse:
    """Same trade-offs as get_clip_video above, for the (much smaller) thumbnail."""
    clip = _get_owned_clip(db, user, clip_id)
    if not clip.thumbnail_key:
        raise ApiError(404, "not_found", "Clip has no thumbnail yet")
    local_path = storage.get_local_path(
        clip.thumbnail_key, download_to=os.path.join(tempfile.gettempdir(), f"clip-thumb-{clip.id}.jpg")
    )
    return FileResponse(local_path, media_type="image/jpeg", filename=f"{clip.id}.jpg")


def _get_owned_clip(db: Session, user: User, clip_id: uuid.UUID) -> RenderedClip:
    clip = db.get(RenderedClip, clip_id)
    if clip is None:
        raise ApiError(404, "not_found", "Clip not found")
    if clip.stream_job.user_id != user.id:
        raise ApiError(403, "forbidden", "You do not have access to this clip")
    return clip
