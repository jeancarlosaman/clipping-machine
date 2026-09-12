"""POST/GET /api/v1/stream-jobs -- architecture doc §6."""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUserDep, DbDep
from app.api.errors import ApiError
from app.core.config import STT_LOCAL_MODEL_SIZES, settings
from app.core.queue import enqueue
from app.core.storage import new_object_key, storage
from app.db.models import RenderedClip, StreamJob, User
from app.schemas import LayoutRegionsRequest, MarkedRect, RenderedClipOut, StreamJobOut
from app.workers import ingest
from app.workers.common import audio_object_key, run_subprocess

router = APIRouter(prefix="/api/v1/stream-jobs", tags=["stream-jobs"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

# See StreamJob.camera_layout_mode's docstring for what each value means.
# "auto" is accepted as an explicit input alias for "use the default
# heuristic" (stored as None, same as not sending the field at all) so the
# dev console/API caller doesn't need to know None is the sentinel.
CAMERA_LAYOUT_MODES = ("auto", "single_crop", "split_reaction", "fit_frame")
# Which side of the frame a 9:16 crop should favour, for the layouts that
# crop at all -- see StreamJob.crop_bias. "center" is accepted explicitly
# (and stored as-is, unlike camera_layout_mode's "auto") because it is a
# real instruction, not a default: it means "center it and do NOT follow a
# detected face", which is different from leaving this unset.
CROP_BIASES = ("left", "center", "right")


def _parse_marked_rect(raw: str | None, field: str) -> dict | None:
    """A framing region submitted as a JSON string in multipart form data.

    The upload form has to send these as strings (multipart has no nested
    objects), so they arrive as e.g. '{"x":0.72,"y":0.6,"w":0.26,"h":0.36}'.
    Validated through the same MarkedRect model the JSON endpoint uses, so
    the two paths cannot drift apart on what counts as a valid box.

    Returns None for absent/blank, which means "not marked" -- the same
    thing the column's default means.
    """
    if raw is None or not raw.strip():
        return None
    try:
        rect = MarkedRect.model_validate(json.loads(raw))
    except Exception as exc:
        raise ApiError(400, f"invalid_{field}", f"{field} must be a JSON object with x/y/w/h in 0..1: {exc}")
    if rect.x + rect.w > 1.0 or rect.y + rect.h > 1.0:
        raise ApiError(
            400, f"invalid_{field}",
            f"The {field} box extends past the edge of the frame "
            f"(x+w={rect.x + rect.w:.3f}, y+h={rect.y + rect.h:.3f}; both must be <= 1.0)",
        )
    return {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h}


# Each entry: (json key, python type, min, max). The bounds are the same
# ones documented on the matching settings.* field -- notably caption_margin_v
# must not drop below 50, which is where the caption band starts colliding
# with TikTok's own bottom UI. A preview slider in a browser is trivially
# bypassed, so the real limit has to live here.
STYLE_OVERRIDE_BOUNDS = {
    "split_facecam_fraction": (float, 0.2, 0.8),
    "caption_font_size": (int, 5, 20),
    "caption_margin_v": (int, 50, 200),
    "caption_max_chars": (int, 20, 200),
    "title_font_size": (int, 6, 30),
    "title_margin_v": (int, 30, 200),
}


def _parse_style_overrides(raw: str | None) -> dict | None:
    """Per-job burn-in styling, submitted as a JSON string in the upload form.

    Unknown keys are rejected rather than ignored: a typo'd key would
    otherwise be silently dropped and the creator would spend a render
    wondering why nothing changed.
    """
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ApiError(400, "invalid_style_overrides", f"style_overrides must be a JSON object: {exc}")
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_style_overrides", "style_overrides must be a JSON object")

    unknown = sorted(set(data) - set(STYLE_OVERRIDE_BOUNDS))
    if unknown:
        raise ApiError(
            400, "invalid_style_overrides",
            f"Unknown style keys: {unknown}. Allowed: {sorted(STYLE_OVERRIDE_BOUNDS)}",
        )

    cleaned: dict = {}
    for key, value in data.items():
        kind, low, high = STYLE_OVERRIDE_BOUNDS[key]
        try:
            typed = kind(value)
        except (TypeError, ValueError):
            raise ApiError(400, "invalid_style_overrides", f"{key} must be a {kind.__name__}")
        if not (low <= typed <= high):
            raise ApiError(400, "invalid_style_overrides", f"{key} must be between {low} and {high}")
        cleaned[key] = typed
    return cleaned or None


def _validate_job_overrides(
    *,
    max_clips: int | None,
    min_score_threshold: float | None,
    min_clip_seconds: float | None,
    max_clip_seconds: float | None,
    stt_model_size: str | None,
    camera_layout_mode: str | None,
    crop_bias: str | None,
    facecam_rect: str | None = None,
    gameplay_rect: str | None = None,
    style_overrides: str | None = None,
) -> dict:
    """Validates the optional per-job overrides a creator can set at upload
    time. Raises ApiError(400, ...) on anything out of bounds -- these bounds
    exist so a bad/malicious value can't produce a pathological job (e.g.
    max_clips=10000, or a 0-second minimum clip length), per the project's
    "hard safety limits" requirement, not just to be pedantic about types.
    Returns a dict of the validated values (still None where not provided),
    ready to pass straight into StreamJob(**overrides).
    """
    if max_clips is not None and not (1 <= max_clips <= settings.hard_max_clips_per_job):
        raise ApiError(
            400, "invalid_max_clips",
            f"max_clips must be between 1 and {settings.hard_max_clips_per_job}",
        )
    if min_score_threshold is not None and not (0.0 <= min_score_threshold <= 10.0):
        raise ApiError(400, "invalid_min_score_threshold", "min_score_threshold must be between 0 and 10")
    if min_clip_seconds is not None and min_clip_seconds < settings.min_clip_seconds_floor:
        raise ApiError(
            400, "invalid_min_clip_seconds",
            f"min_clip_seconds must be at least {settings.min_clip_seconds_floor}",
        )
    if max_clip_seconds is not None and max_clip_seconds > settings.max_clip_seconds_ceiling:
        raise ApiError(
            400, "invalid_max_clip_seconds",
            f"max_clip_seconds must be at most {settings.max_clip_seconds_ceiling}",
        )
    # Compare against the EFFECTIVE value on each side, not just what this
    # request happened to submit -- a real bug found in production: a job
    # explicitly overrode min_clip_seconds=35 while leaving max_clip_seconds
    # unset (falling back to a .env SEGMENT_MAX_CLIP_SECONDS that had been
    # left at 10 from earlier short-test-video tuning). The old version of
    # this check only fired when BOTH were provided in the same request, so
    # this inverted-but-not-both-explicit combination sailed straight
    # through and silently broke app.core.segmentation_logic
    # .build_candidate_windows/_sliding_subwindows downstream: with
    # min_len > max_len, `step = max(max_len/3, min_len)` collapses to
    # min_len, and _sliding_subwindows still runs, so every clip came out
    # ~max_len long (here, ~10s) spaced ~min_len apart (~35s) -- exactly the
    # "all clips are 9-10 seconds despite min_clip_seconds=35" symptom a
    # user hit for real. Comparing effective values here, and the same
    # comparison again defensively in app/workers/segmentation.py, closes
    # both the API-input gap and the "job somehow reached segmentation with
    # an inverted range anyway" gap (an old row, a manual DB edit, or a
    # future caller that bypasses this endpoint).
    effective_min = min_clip_seconds if min_clip_seconds is not None else settings.segment_min_clip_seconds
    effective_max = max_clip_seconds if max_clip_seconds is not None else settings.segment_max_clip_seconds
    if effective_min >= effective_max:
        raise ApiError(
            400, "invalid_clip_length_range",
            f"min_clip_seconds ({effective_min}) must be less than max_clip_seconds ({effective_max}) -- "
            "note this compares against the configured default for whichever one wasn't set on this request",
        )
    if stt_model_size is not None and stt_model_size not in STT_LOCAL_MODEL_SIZES:
        raise ApiError(
            400, "invalid_stt_model_size",
            f"stt_model_size must be one of {list(STT_LOCAL_MODEL_SIZES)}",
        )
    if camera_layout_mode is not None and camera_layout_mode not in CAMERA_LAYOUT_MODES:
        raise ApiError(
            400, "invalid_camera_layout_mode",
            f"camera_layout_mode must be one of {list(CAMERA_LAYOUT_MODES)}",
        )
    if crop_bias is not None and crop_bias not in CROP_BIASES:
        raise ApiError(
            400, "invalid_crop_bias",
            f"crop_bias must be one of {list(CROP_BIASES)}",
        )
    return {
        "max_clips": max_clips if max_clips is not None else settings.default_max_clips_per_job,
        "min_score_threshold": min_score_threshold,
        "min_clip_seconds": min_clip_seconds,
        "max_clip_seconds": max_clip_seconds,
        "stt_model_size": stt_model_size,
        # "auto" is just the explicit way to say "store null" -- see
        # CAMERA_LAYOUT_MODES' comment above.
        "camera_layout_mode": camera_layout_mode if camera_layout_mode != "auto" else None,
        "crop_bias": crop_bias,
        # Marked at upload time in the browser, from a frame of the file the
        # creator just picked -- so the very first render already frames
        # correctly instead of needing the job to exist first.
        "facecam_rect": _parse_marked_rect(facecam_rect, "facecam_rect"),
        "gameplay_rect": _parse_marked_rect(gameplay_rect, "gameplay_rect"),
        "style_overrides": _parse_style_overrides(style_overrides),
    }


@router.post("", response_model=StreamJobOut, status_code=201)
def create_stream_job_from_upload(
    file: UploadFile = File(...),
    max_clips: int | None = Form(default=None),
    min_score_threshold: float | None = Form(default=None),
    min_clip_seconds: float | None = Form(default=None),
    max_clip_seconds: float | None = Form(default=None),
    stt_model_size: str | None = Form(default=None),
    camera_layout_mode: str | None = Form(default=None),
    crop_bias: str | None = Form(default=None),
    facecam_rect: str | None = Form(default=None),
    gameplay_rect: str | None = Form(default=None),
    style_overrides: str | None = Form(default=None),
    db: Session = DbDep,
    user: User = CurrentUserDep,
) -> StreamJob:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ApiError(
            400, "unsupported_file_type",
            f"Unsupported file extension '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    overrides = _validate_job_overrides(
        max_clips=max_clips,
        min_score_threshold=min_score_threshold,
        min_clip_seconds=min_clip_seconds,
        max_clip_seconds=max_clip_seconds,
        stt_model_size=stt_model_size,
        camera_layout_mode=camera_layout_mode,
        crop_bias=crop_bias,
        facecam_rect=facecam_rect,
        gameplay_rect=gameplay_rect,
        style_overrides=style_overrides,
    )

    # Stream to a temp file first so we can enforce the size cap without
    # buffering the whole upload in memory, then hand it to storage.
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp_path = tmp.name
        total_bytes = 0
        while chunk := file.file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > settings.max_upload_bytes:
                Path(tmp_path).unlink(missing_ok=True)
                raise ApiError(
                    413, "file_too_large",
                    f"File exceeds the {settings.max_upload_bytes} byte upload limit",
                )
            tmp.write(chunk)

    try:
        object_key = new_object_key(prefix="raw", extension=extension.lstrip("."))
        storage.put_file(tmp_path, object_key)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    job = StreamJob(
        user_id=user.id,
        source_type="upload",
        raw_object_key=object_key,
        status="queued",
        **overrides,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Duration isn't known yet (that's what ingest's own ffprobe call
    # determines) -- job_timeout=1800 is a flat, generous ceiling rather
    # than RQ's 180s default, which is nowhere near enough for a real VOD.
    # See app.workers.common.estimate_job_timeout_seconds for how every
    # later stage scales its own timeout once duration_seconds is known.
    enqueue("ingest", ingest.run, str(job.id), on_failure=ingest.on_failure, job_timeout=1800)

    return job


@router.get("", response_model=list[StreamJobOut])
def list_stream_jobs(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, le=100),
    db: Session = DbDep,
    user: User = CurrentUserDep,
) -> list[StreamJob]:
    stmt = select(StreamJob).where(StreamJob.user_id == user.id)
    if status_filter:
        stmt = stmt.where(StreamJob.status == status_filter)
    stmt = stmt.order_by(StreamJob.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))


@router.get("/{stream_job_id}", response_model=StreamJobOut)
def get_stream_job(stream_job_id: uuid.UUID, db: Session = DbDep, user: User = CurrentUserDep) -> StreamJob:
    return _get_owned_job(db, user, stream_job_id)


@router.delete("/{stream_job_id}", status_code=204, response_model=None)
def delete_stream_job(stream_job_id: uuid.UUID, db: Session = DbDep, user: User = CurrentUserDep) -> None:
    """Deletes a stream_job and everything under it -- transcript,
    candidate_segments, rendered_clips, upload_tasks all cascade at the DB
    level (ondelete="CASCADE" on each FK, see app.db.models). Also cleans
    up the actual storage objects (raw video, extracted audio, every
    rendered clip's video/thumbnail), since a DB-only delete would just
    leak files -- fine for a couple of dev test uploads, not fine as a
    real habit.

    No guard against deleting a job that's mid-pipeline: an in-flight
    worker job for a deleted stream_job_id already handles "row not found"
    as a clean no-op everywhere in this codebase (same pattern every
    worker's `run()` uses), so this is safe to call at any status.
    """
    job = _get_owned_job(db, user, stream_job_id)

    clips = list(db.scalars(select(RenderedClip).where(RenderedClip.stream_job_id == job.id)))
    for clip in clips:
        if clip.object_key:
            storage.delete(clip.object_key)
        if clip.thumbnail_key:
            storage.delete(clip.thumbnail_key)

    storage.delete(job.raw_object_key)
    storage.delete(audio_object_key(str(job.id)))

    db.delete(job)
    db.commit()


@router.get("/{stream_job_id}/clips", response_model=list[RenderedClipOut])
def list_stream_job_clips(
    stream_job_id: uuid.UUID, db: Session = DbDep, user: User = CurrentUserDep
) -> list[RenderedClip]:
    _get_owned_job(db, user, stream_job_id)  # 404s / ownership-checks first
    stmt = (
        select(RenderedClip)
        .where(RenderedClip.stream_job_id == stream_job_id)
        .order_by(RenderedClip.created_at.desc())
    )
    return list(db.scalars(stmt))


@router.get("/{stream_job_id}/frame")
def get_stream_job_frame(
    stream_job_id: uuid.UUID,
    at_seconds: float = Query(default=0.0, ge=0.0, description="Timestamp to grab, in source seconds"),
    db: Session = DbDep,
    user: User = CurrentUserDep,
) -> FileResponse:
    """One JPEG frame from the job's source video, for marking the facecam.

    Exists so a creator can point at the facecam on a real frame of their
    own VOD instead of relying on face detection to find it (see
    StreamJob.facecam_rect). The frame is returned at full source
    resolution -- the dev console scales it for display and sends back
    NORMALIZED coordinates, so this endpoint never has to agree with the
    browser about pixel sizes.

    `-ss` before `-i` so ffmpeg seeks by keyframe rather than decoding from
    zero -- effectively instant even hours into a long VOD, at the cost of
    landing on the nearest keyframe rather than the exact timestamp. That
    trade is right here: this frame is a visual reference for drawing a
    box, not a frame-accurate extract.
    """
    job = _get_owned_job(db, user, stream_job_id)

    source = storage.get_local_path(
        job.raw_object_key,
        download_to=str(Path(tempfile.gettempdir()) / f"job-source-{job.id}"),
    )
    out_path = Path(tempfile.gettempdir()) / f"job-frame-{job.id}-{at_seconds:.2f}.jpg"

    result = run_subprocess(
        [
            "ffmpeg", "-y", "-ss", str(at_seconds), "-i", source,
            "-frames:v", "1", "-q:v", "3", str(out_path),
        ],
        timeout_seconds=120,
    )
    if result.returncode != 0 or not out_path.exists():
        raise ApiError(
            422, "frame_extract_failed",
            f"Could not read a frame at {at_seconds}s -- is the timestamp past the end of the video?",
        )

    # Deleted after the response is actually sent: FileResponse streams the
    # file AFTER this function returns, so unlinking here would race it.
    return FileResponse(
        out_path,
        media_type="image/jpeg",
        filename=f"{job.id}-{at_seconds:.2f}.jpg",
        background=BackgroundTask(lambda: out_path.unlink(missing_ok=True)),
    )


@router.put("/{stream_job_id}/layout-regions", response_model=StreamJobOut)
def set_stream_job_layout_regions(
    stream_job_id: uuid.UUID,
    payload: LayoutRegionsRequest,
    db: Session = DbDep,
    user: User = CurrentUserDep,
) -> StreamJob:
    """Record (or clear) the hand-marked facecam and content regions.

    These replace guessing. `facecam` forces the split layout and fixes its
    top panel; `gameplay` fixes the bottom panel, or -- when nothing is
    split -- the region a single 9:16 crop is taken from. Face detection,
    classify_reaction_layout and crop_bias are all bypassed for whichever
    region is marked, because each of those exists only to guess at what is
    being stated here directly.

    Takes effect on the NEXT render of this job's clips; clips already
    rendered keep the framing they were rendered with. The useful moment is
    right after upload, while transcription and segmentation are still
    running -- rendering is minutes away, so a mark made then lands in time
    for the first render with nothing re-done.
    """
    job = _get_owned_job(db, user, stream_job_id)
    provided = payload.model_fields_set

    for field, column in (("facecam", "facecam_rect"), ("gameplay", "gameplay_rect")):
        if field not in provided:
            continue  # omitted entirely -- leave this region untouched
        rect = getattr(payload, field)
        if rect is None:
            setattr(job, column, None)
            continue
        # Per-field bounds come from the schema; this is the one constraint
        # that spans fields -- a box may not run off the edge of the frame.
        if rect.x + rect.w > 1.0 or rect.y + rect.h > 1.0:
            raise ApiError(
                400, "invalid_layout_region",
                f"The {field} box extends past the edge of the frame "
                f"(x+w={rect.x + rect.w:.3f}, y+h={rect.y + rect.h:.3f}; both must be <= 1.0)",
            )
        setattr(job, column, {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h})

    db.commit()
    db.refresh(job)
    return job


# Cancelling a job that has already finished (or already failed) is
# meaningless, and silently "succeeding" would make the console lie about
# what happened -- so these are refused rather than accepted as a no-op.
TERMINAL_JOB_STATUSES = (
    "ready_for_review",
    "archived",
    "cancelled",
    "failed_ingest",
    "failed_transcription",
    "failed_segmentation",
    "failed_scoring",
    "failed_rendering",
)


@router.post("/{stream_job_id}/cancel", response_model=StreamJobOut)
def cancel_stream_job(
    stream_job_id: uuid.UUID,
    db: Session = DbDep,
    user: User = CurrentUserDep,
) -> StreamJob:
    """Stop a running job at its next stage boundary.

    Cooperative, not a kill: this only sets the status. Each worker checks it
    at the top of its stage (app.workers.common.job_is_cancelled) and returns
    without enqueuing the next one, so whatever is mid-flight right now --
    a transcription pass, an ffmpeg render -- finishes first. The pipeline
    then stops with nothing half-written: no truncated mp4, no orphaned temp
    directory, no row left in an impossible state.

    In practice that means cancelling during transcription of a long VOD can
    take a few minutes to visibly stop. That is the deliberate trade; the
    alternative (killing the subprocess) saves those minutes and gives up the
    cleanup guarantee.

    Queued-but-not-started work stops immediately, since those jobs hit the
    check before doing anything.
    """
    job = _get_owned_job(db, user, stream_job_id)

    if job.status in TERMINAL_JOB_STATUSES:
        raise ApiError(
            409, "not_cancellable",
            f"This job is already finished (status: {job.status}) -- nothing to cancel.",
        )

    job.status = "cancelled"
    db.commit()
    db.refresh(job)
    return job


def _get_owned_job(db: Session, user: User, stream_job_id: uuid.UUID) -> StreamJob:
    job = db.get(StreamJob, stream_job_id)
    if job is None:
        raise ApiError(404, "not_found", "Stream job not found")
    if job.user_id != user.id:
        raise ApiError(403, "forbidden", "You do not have access to this stream job")
    return job
