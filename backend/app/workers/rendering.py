"""Rendering worker -- architecture doc §7.

Trigger:  a candidate_segment reaching status='selected' -- enqueued
          per-clip by app.workers.scoring.run, which also creates the
          RenderedClip row (status='pending') this worker updates.
Input:    candidate_segment_id
Output:   the candidate's RenderedClip row updated with object_key
          (rendered 9:16 mp4), thumbnail_key (jpg), duration_seconds, and
          status.
State:    pending -> rendering -> rendered | failed (per clip). The
          "all clips terminal -> job.status=ready_for_review" aggregate
          transition is intentionally NOT implemented here -- see "Next
          steps" in the README; it needs to run after whichever clip
          finishes last, which is a different (small) piece of logic.
Retries:  3x per clip, isolated -- one clip's ffmpeg failure must not block
          its siblings, which is why this worker takes a
          candidate_segment_id rather than a stream_job_id (see
          app.workers.scoring.run, which enqueues one job per clip).

ffmpeg does the real work: cut the candidate window from the raw video
(-ss/-t), then one of two layouts (app.core.rendering_logic), scaled to a
fixed 1080x1920 output, with caption burn-in via ffmpeg's `subtitles`
filter (libass) from an SRT built out of the transcript slice covering the
window:

- **Single crop** (the default, and the fallback whenever face detection
  finds nothing): a 9:16 crop, centered on a detected face when
  app.core.face_detect finds one, otherwise centered on the frame -- see
  compute_crop_offset. Fine for "IRL"/single-camera content, where the crop
  just needs to keep the subject in frame.
- **Split reaction layout**: when the detected face looks like a small,
  corner-positioned webcam box rather than someone filling the frame (see
  app.core.rendering_logic.classify_reaction_layout) -- i.e. a streamer
  reacting to separate gameplay/video content -- the top
  half of the output is a tight zoom on the facecam
  (compute_face_zoom_crop) and the bottom half is a plain crop of the full
  frame, stacked with ffmpeg's `vstack` filter
  (build_split_reaction_filtergraph). This doesn't attempt to detect and
  exclude the webcam box's exact rectangle from the bottom half -- that
  would need real rectangle detection this MVP doesn't have -- so the
  bottom half may still show a small sliver of it; a known, flagged
  limitation, not a bug.

Both layouts are still a single *static* choice per clip, not
motion-tracked/smart reframing that follows movement within the clip --
that remains an explicit post-MVP trade-off, see architecture doc §8.

Text burn-in (both stages below) is the step most likely to behave
differently across ffmpeg builds (needs libass compiled in) -- if it
fails, this worker retries once without either burned-in text stage rather
than losing the whole clip over it, and flags the clip accordingly
('captions_failed' / 'title_failed') so a reviewer notices instead of
silently getting a clip missing text it should have had.

- **Transcript captions**: burned in from an SRT built out of the
  transcript slice covering the window, as above.
- **Clickbait title banner**: a second `subtitles` burn-in stage, chained
  after the transcript captions, showing a short punchy hook
  (`app.core.rendering_logic.build_title_srt`) top-center for the clip's
  whole duration (`_TITLE_STYLE` below -- bigger/bolder than the caption
  style, `Alignment=8`). The title text comes from
  `app.core.caption_generation.generate_caption_annotation` (LLM-written,
  with a deterministic heuristic fallback -- see that module), called
  **synchronously here, before the filtergraph is built** -- unlike
  hashtags/caption/explanation (metadata only), the title has to be known
  before this point since it gets burned into the actual pixels; there's
  no "generate it after and patch the video" option short of a full
  re-render. Toggle with `ENABLE_CLIP_TITLE_OVERLAY` (default `true`),
  independent of `ENABLE_LLM_CAPTIONS` (which only controls whether the
  title text itself is LLM-written vs. the heuristic fallback, not whether
  it gets burned in at all).

**Thumbnail selection**: rather than always grabbing whatever frame sits at
a fixed t=0.1s offset into the rendered clip (the original behavior), a
handful of frames are sampled across the clip and
app.core.thumbnail_selection.pick_best_frame ranks them by sharpness
(Laplacian variance -- penalizes motion blur/fast pans) and whether a face
is visible (reusing app.core.face_detect's Haar cascade, already a
dependency), excluding near-black/near-white "dead" frames (almost always a
transition) when a better option exists. Falls back to the original
fixed-frame extract if sampling/scoring finds nothing usable -- never a
worse thumbnail than before this feature existed, just sometimes not a
better one either.

Once the annotation is generated (title/hashtags/caption/explanation, one
call), it's persisted to `candidate_segments.llm_annotation` /
`rendered_clips.caption_text` in the same success block that marks the
clip `rendered` -- there is no separate after-render async enqueue
anymore (there used to be, before the title needed to exist pre-render;
see `app.workers.caption_generation`'s docstring). That worker is kept
around as a manual regenerate path -- e.g. to redo an annotation without
re-rendering the video -- but is no longer auto-invoked by this one.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid

from app.core.caption_generation import generate_caption_annotation
from app.core.config import settings
from app.core.face_detect import estimate_face_profile
from app.core.rendering_logic import (
    crop_from_marked_rect,
    marked_rect_pixels,
    append_subtitles_stage,
    build_concat_prefix,
    build_fit_frame_filtergraph,
    build_multipart_srt,
    build_single_crop_filtergraph,
    build_split_reaction_filtergraph,
    build_srt,
    build_title_srt,
    classify_reaction_layout,
    compute_crop_offset,
    compute_face_zoom_crop,
    compute_vertical_crop,
    normalize_parts,
    parts_duration,
    retarget_source_label,
)
from app.core.storage import new_object_key, storage
from app.core.thumbnail_selection import pick_best_frame
from app.db.models import CandidateSegment, RenderedClip, StreamJob, Transcript
from app.workers.common import candidate_job_is_cancelled, db_session, logger, run_subprocess

# Fractions of a clip's own [start, end] window to sample for face
# detection -- not the whole source video, just this clip's slice, so the
# extra ffmpeg frame extracts stay cheap regardless of how long the source
# VOD is. Three points rather than one so a webcam that's briefly occluded
# or a subject who blinks/looks away doesn't wipe out the whole estimate.
_FACE_SAMPLE_FRACTIONS = (0.25, 0.5, 0.75)

# Fractions of the *rendered* clip (post-crop/caption burn-in, since that's
# what a reviewer/viewer actually sees) to sample as thumbnail candidates --
# see app.core.thumbnail_selection.pick_best_frame for how the best one is
# chosen. Avoids the very start/end (0.1/0.95 territory) deliberately: a
# clip boundary is exactly where a scene cut, fade, or mid-word freeze frame
# is most likely to land, which is also why the *old* fixed "grab frame at
# 0.1s" behavior this replaces was a real problem worth fixing. Six points
# is enough variety to usually catch at least one sharp, non-transition,
# preferably face-visible frame without extracting so many that thumbnail
# generation becomes the slow part of rendering one clip.
_THUMBNAIL_SAMPLE_FRACTIONS = (0.12, 0.28, 0.44, 0.6, 0.76, 0.9)

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

# Readable on a phone-sized vertical video: compact, sitting low in the
# bottom third, black outline for contrast against any background. Values
# were tuned empirically against a real 1080x1920 render (not guessed from
# the ASS spec) -- ffmpeg's `subtitles` filter font sizing depends on
# libass's actual script-resolution handling, not just the raw FontSize
# number, so what looks right has to be checked against real output.
# FontSize=20/MarginV=120 (the original default) rendered oversized and sat
# closer to mid-frame than the bottom -- FontSize=10/MarginV=60 reads
# clearly at arm's length and sits where short-form captions are expected.
# MarginV=40 (was 60): moved lower per user feedback after reviewing a
# render -- re-verified empirically the same way (synthetic 1080x1920
# render + measuring the actual white-pixel rows), landing the caption
# band at roughly 83-86% down the frame (~14% up from the bottom edge),
# vs. ~76-79% (~21% up) at MarginV=60. Still clear of TikTok's own bottom
# UI band (typically the bottom ~15-20%), just closer to the edge than
# before.
# MarginV=28 (was 40): moved lower again per user feedback -- re-verified
# empirically the same way (synthetic 1080x1920 black frame + a real
# libass render + measuring the actual white-pixel rows), landing the
# caption band at ~84-90% down the frame (~10% up from the bottom edge),
# vs. ~80-86% (~14% up) at MarginV=40. This is now MORE overlap risk with
# TikTok's own bottom UI band (still only ever estimated at "typically
# 15-20%" in this codebase, never verified against the real app from this
# environment) than the previous value already carried -- if a real render
# on TikTok shows the caption clipped by or fighting the platform's own
# UI, move this back up (a bigger MarginV number) rather than pushing it
# any further down.
# Sizes and margins now live in app/core/config.py (caption_font_size,
# caption_margin_v, title_font_size, title_margin_v) so they are tunable
# from .env without a code change -- each carries the measured numbers and
# the safe floor in its comment there. Per-CLIP styling is still a
# later-research item; this is one style for every clip.
# MarginV=55 (was 28) -- 2026-09-11. The two previous "move it lower"
# adjustments (60 -> 40 -> 28) walked the caption band INTO TikTok's own
# bottom UI. Researched TikTok's safe zones and then measured this exact
# style with a real libass render on a 1080x1920 frame:
#     MarginV=28  band at y=1611-1732, only 188px above the frame bottom
#     MarginV=40  band at y=1531-1652, 268px
#     MarginV=50  band at y=1464-1586, 334px  (exactly at the limit)
#     MarginV=55  band at y=1431-1552, 368px  <- chosen
#     MarginV=60  band at y=1397-1519, 401px  (the original value)
# TikTok's organic bottom overlay -- @username, the caption/description and
# the audio marquee -- occupies roughly the bottom 334px (17.4%). At
# MarginV=28 the ENTIRE caption band sat inside that, i.e. underneath
# TikTok's own text. 55 clears it with ~34px to spare while still sitting
# as low as the safe zone allows, which is what the "lower" feedback was
# actually after. Do not push this below 50 without re-measuring: the
# preview in the dev console shows the raw clip, which is exactly why this
# regression was invisible for three iterations.
def _caption_style(style: dict) -> str:
    return (
        f"FontSize={style['caption_font_size']},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=1.5,Shadow=0,Alignment=2,MarginV={style['caption_margin_v']}"
    )

# The clickbait title banner -- bigger and bolder than the transcript
# captions above (this is the "headline" a viewer reads first, not
# dialogue), top-center with a thicker outline for a bit more pop.
# White-on-black-outline like the captions rather than a filled color box:
# keeps this to one MVP-default style (no per-clip customization yet, same
# trade-off as _CAPTION_STYLE) while staying legible over any background.
#
# `Alignment=6`, NOT 8 -- verified empirically (rendered real frames and
# measured actual pixel positions), not guessed from the ASS v4+ spec.
# ffmpeg's `subtitles` filter converts a plain SRT into ASS internally, and
# that conversion's `force_style` Alignment field turned out to follow the
# legacy SSA v4 numbering (5/6/7 = top-left/center/right, 9/10/11 =
# mid-left/center/right, 1/2/3 = bottom-left/center/right -- 4 and 8 are
# unused in that scheme) rather than the ASS v4+ numpad-style numbering
# (1-9, 8 = top-middle) `force_style` options otherwise resemble.
# Alignment=8 rendered the banner across the vertical MIDDLE of the frame,
# not the top -- a real, non-obvious gotcha, not a typo; kept this comment
# so nobody "corrects" it back to 8. _CAPTION_STYLE's Alignment=2
# (bottom-center) happens to mean the same thing under both numbering
# schemes, which is why that one never needed this.
#
# `MarginV=10` (small, unlike the caption's 60) -- FontSize/MarginV here
# are also interpreted against libass's internal default script
# resolution, not the real 1080x1920 output (same class of issue noted in
# _CAPTION_STYLE's own comment), so this was tuned empirically against a
# real render, not derived from the numbers alone.
# FontSize=16 -> 13 (per user feedback that the banner took up too much of
# the top of the frame): re-verified empirically the same way (synthetic
# 1080x1920 black frame + a real libass render + measuring the actual
# white-pixel rows). This is a bigger win than "10% smaller text" implies
# -- at FontSize=16, a typical ~30-40 char title (e.g. "He Just Called Out
# The Community") already wrapped to 2 lines at ~281px tall (14.6% of
# frame height); at FontSize=13 the same text fits on ONE line at ~141px
# (7.3%) -- roughly half the vertical footprint, not just a smaller font.
# Even a full MAX_TITLE_CHARS=70 title, which still wraps to 2 lines at
# FontSize=13, comes out to ~230px (12.0%) -- still noticeably smaller
# than the old default's typical (non-worst-case) footprint. Go smaller
# than 13 only with a fresh empirical check -- legibility "at arm's length
# on a phone" (the same bar _CAPTION_STYLE's own comment sets) isn't
# something to guess past.
# MarginV=35 (was 10) -- 2026-09-11, same safe-zone audit as
# _CAPTION_STYLE above. TikTok's top chrome (search icon, LIVE badge,
# Following/For You tabs) covers roughly the top 200px (10.4%). Measured
# with a real libass render, worst case being a full MAX_TITLE_CHARS=70
# title that wraps to two lines:
#     MarginV=10  title at y=80-326   -- top 120px hidden behind TikTok UI
#     MarginV=20  title at y=147-392  -- still clipped
#     MarginV=30  title at y=214-459  -- clears by 14px
#     MarginV=35  title at y=247-492  -- chosen, clears by 47px
# The banner is the first thing a viewer is supposed to read, so having its
# top third behind the platform's own navigation was costing exactly the
# hook the title exists to deliver.
def _title_style(style: dict) -> str:
    return (
        f"FontSize={style['title_font_size']},Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,Alignment=6,MarginV={style['title_margin_v']}"
    )


def resolve_style(overrides: dict | None) -> dict:
    """This job's burn-in styling: settings.* defaults with any per-job
    override laid over the top.

    Values are already range-checked by the API (STYLE_OVERRIDE_BOUNDS), but
    a row could also have been written by an older client or edited by hand,
    so unknown keys are ignored here rather than trusted -- a bad style
    should cost the clip its styling, never the render.
    """
    resolved = {
        "split_facecam_fraction": settings.split_facecam_fraction,
        "caption_font_size": settings.caption_font_size,
        "caption_margin_v": settings.caption_margin_v,
        "caption_max_chars": settings.caption_max_chars,
        "title_font_size": settings.title_font_size,
        "title_margin_v": settings.title_margin_v,
    }
    for key, value in (overrides or {}).items():
        if key in resolved and isinstance(value, (int, float)):
            resolved[key] = value
    return resolved


def _probe_video_dimensions(local_path: str) -> tuple[int, int]:
    result = run_subprocess(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            local_path,
        ]
    )
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("no video stream found")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _probe_has_audio(local_path: str) -> bool:
    """Whether the source has any audio stream at all.

    Only needed for the stitched multi-part path: the single-window render
    below maps audio with ffmpeg's optional `0:a?` and simply produces a
    silent clip when there's none, but a filtergraph that *references*
    `[0:a]` fails the whole command when that stream doesn't exist -- so
    the concat prefix has to know in advance whether to build an audio
    branch (see app.core.rendering_logic.build_concat_prefix). Treats a
    probe failure as "no audio": a silent stitched clip is a far better
    outcome than a failed render.
    """
    try:
        result = run_subprocess(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type", "-of", "json", local_path,
            ]
        )
        return bool(json.loads(result.stdout).get("streams"))
    except Exception:
        return False


def _render(
    local_video_path: str,
    out_path: str,
    start: float,
    end: float,
    filtergraph: str,
    srt_path: str | None,
    title_srt_path: str | None,
    style: dict,
    parts: list[tuple[float, float]] | None = None,
) -> None:
    # `style` is REQUIRED, not defaulted: caption/title sizing became
    # per-job (StreamJob.style_overrides), and a default here would let a
    # caller silently render with the wrong styling instead of failing
    # loudly at the call site.
    # -filter_complex (not the simpler -vf) because the split-reaction
    # layout needs a branching graph (two crops off the same source frame,
    # merged with vstack) that -vf's linear-chain-only syntax can't express
    # -- see app.core.rendering_logic's filtergraph builders. The
    # single-crop layout's graph is just as valid under -filter_complex, so
    # this is one code path for both rather than two. filter_complex
    # outputs aren't auto-selected like a plain -vf's are, so the video
    # stream must be explicitly -map'd; `0:a?` marks audio optional so this
    # doesn't fail outright on a (rare, but possible) source with no audio
    # track.
    graph = filtergraph
    if srt_path is not None:
        graph = append_subtitles_stage(graph, srt_path, _caption_style(style))
    if title_srt_path is not None:
        # Chained as its own subtitles stage (not merged into the same SRT
        # as the transcript captions) so it can use a completely different
        # style/position (_TITLE_STYLE's top alignment vs _CAPTION_STYLE's
        # bottom) -- append_subtitles_stage already handles relabeling
        # [vout]->[vpre] correctly however many times it's chained.
        graph = append_subtitles_stage(graph, title_srt_path, _title_style(style))

    if parts:
        # Stitched multi-part clip: cut each part out with trim/atrim and
        # concat them into one stream FIRST, then point the layout graph at
        # that stream instead of the raw input. Every layout keeps working
        # unchanged because none of them knows where its source frames came
        # from (see build_concat_prefix/retarget_source_label).
        #
        # No -ss/-t here on purpose: the trim filters already define exactly
        # what plays, and an outer -ss would shift the timestamps the trims
        # are expressed in, silently cutting the wrong material. Audio is
        # mapped from the concat branch when the source actually has audio,
        # and left unmapped otherwise -- `0:a?` wouldn't help here, since
        # after concat the audio no longer corresponds to the raw input's
        # timeline at all.
        has_audio = _probe_has_audio(local_video_path)
        graph = f"{build_concat_prefix(parts, include_audio=has_audio)};{retarget_source_label(graph, 'vsrc')}"
        audio_map = ["-map", "[asrc]"] if has_audio else []
        run_subprocess(
            [
                "ffmpeg", "-y",
                "-i", local_video_path,
                "-filter_complex", graph,
                "-map", "[vout]",
                *audio_map,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path,
            ],
            timeout_seconds=600,
        )
        return

    run_subprocess(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", local_video_path,
            "-t", f"{max(end - start, 0.01):.3f}",
            "-filter_complex", graph,
            "-map", "[vout]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ],
        timeout_seconds=600,
    )


def _extract_thumbnail_candidates(video_path: str, duration: float, tmp_dir: str) -> list[str]:
    """Sample a handful of frames across the rendered clip for
    app.core.thumbnail_selection.pick_best_frame to choose among. Any
    individual extract failing is skipped, same "best-effort, never block
    the render" posture as _extract_face_sample_frames above."""
    paths = []
    for i, frac in enumerate(_THUMBNAIL_SAMPLE_FRACTIONS):
        timestamp = max(0.0, min(duration * frac, max(duration - 0.05, 0.0)))
        out_path = os.path.join(tmp_dir, f"thumb_candidate_{i}.jpg")
        try:
            run_subprocess(
                ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path, "-frames:v", "1", out_path],
                timeout_seconds=30,
            )
            paths.append(out_path)
        except RuntimeError:
            continue
    return paths


def _extract_thumbnail(video_path: str, thumb_path: str, duration: float, tmp_dir: str) -> None:
    """Picks a good representative frame for the thumbnail instead of
    always grabbing whatever sits at a fixed t=0.1s offset (the original
    behavior, which could just as easily land on a fade transition, a
    mid-blink freeze frame, or a blank loading screen as anything worth
    showing a reviewer). Falls back to that original fixed-frame behavior
    if candidate sampling/scoring fails or finds nothing usable, so this
    can never produce a *worse* thumbnail than before this feature existed
    -- only ffmpeg failing entirely (already handled by this function's
    caller) is a real failure.
    """
    candidates = _extract_thumbnail_candidates(video_path, duration, tmp_dir)
    best_path = None
    if candidates:
        try:
            best_path = pick_best_frame(candidates)
        except Exception as exc:
            logger.warning("rendering.thumbnail_selection_failed", error=str(exc))
            best_path = None
    if best_path:
        shutil.copyfile(best_path, thumb_path)
        return
    run_subprocess(
        ["ffmpeg", "-y", "-i", video_path, "-ss", "0.1", "-frames:v", "1", thumb_path],
        timeout_seconds=60,
    )


def _extract_face_sample_frames(local_video_path: str, start: float, end: float, tmp_dir: str) -> list[str]:
    """A few frame images sampled across this clip's own [start, end]
    window (not the whole source video) for app.core.face_detect to look
    at. `-ss` before `-i` seeks near-instantly to the nearest keyframe, so
    this stays cheap regardless of source video length. Any individual
    extract failing is skipped rather than raised -- face-aware cropping is
    a best-effort enhancement (see settings.enable_face_aware_crop), not a
    required step.
    """
    paths = []
    duration = max(end - start, 0.01)
    for i, frac in enumerate(_FACE_SAMPLE_FRACTIONS):
        timestamp = start + duration * frac
        out_path = os.path.join(tmp_dir, f"face_sample_{i}.jpg")
        try:
            run_subprocess(
                ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", local_video_path, "-frames:v", "1", out_path],
                timeout_seconds=30,
            )
        except RuntimeError:
            continue
        if os.path.exists(out_path):
            paths.append(out_path)
    return paths


def run(candidate_segment_id: str) -> None:
    log = logger.bind(candidate_segment_id=candidate_segment_id, worker="rendering")
    log.info("rendering.start")

    # Cooperative cancel (see app.workers.common.job_is_cancelled): stop at
    # this stage boundary instead of doing the work and enqueuing the next
    # stage. Returning rather than raising keeps this out of the failure
    # path -- a cancelled job is not a failed one, and must not burn retries
    # or trip the on_failure callback.
    if candidate_job_is_cancelled(candidate_segment_id):
        log.info("rendering.cancelled")
        return

    with db_session() as db:
        candidate = db.get(CandidateSegment, uuid.UUID(candidate_segment_id))
        if candidate is None:
            # Not retryable -- the row doesn't exist, retrying changes nothing.
            log.error("rendering.candidate_not_found")
            return

        clip = (
            db.query(RenderedClip)
            .filter(RenderedClip.candidate_segment_id == candidate.id)
            .order_by(RenderedClip.created_at.desc())
            .first()
        )
        if clip is None:
            # scoring.run always creates this row before enqueuing -- a
            # missing row means the job was enqueued some other way (or the
            # row was deleted). Not retryable either way.
            log.error("rendering.clip_not_found")
            return

        clip.status = "rendering"
        clip.retry_count = clip.retry_count + 1
        clip_id = clip.id
        candidate_id = candidate.id

        job = db.get(StreamJob, candidate.stream_job_id)
        raw_object_key = job.raw_object_key
        stream_job_id = job.id
        # Captured here, while the session is open -- the job is detached
        # by the time the caption call below needs it.
        job_user_id = job.user_id
        camera_layout_mode = job.camera_layout_mode  # None ("auto") | single_crop | split_reaction | fit_frame
        crop_bias = job.crop_bias
        facecam_rect = job.facecam_rect
        gameplay_rect = job.gameplay_rect
        style = resolve_style(job.style_overrides)  # None | "left" | "center" | "right" -- see compute_crop_offset

        transcript = db.query(Transcript).filter_by(stream_job_id=job.id).one_or_none()
        transcript_segments = list(transcript.segments) if transcript else []

        start = float(candidate.start_seconds)
        end = float(candidate.end_seconds)
        # None for an ordinary contiguous clip -- every heuristic candidate,
        # and every candidate at all before multi-part existed. When set,
        # start/end still describe the SPAN this clip is drawn from, while
        # `parts` describes what actually plays (see CandidateSegment.parts).
        parts = normalize_parts(candidate.parts)
        score_breakdown = candidate.score_breakdown
        existing_caption = clip.caption_text or ""

    # Real playing time: the sum of a stitched clip's parts, or the plain
    # window length. Drives the burned-in title's duration, the thumbnail
    # sampling range, and the stored duration a reviewer sees -- all of
    # which would be wrong if they used the span for a stitched clip.
    clip_duration = parts_duration(parts) if parts else end - start

    tmp_dir = tempfile.mkdtemp(prefix=f"render-{candidate_segment_id}-")
    try:
        try:
            local_video_path = storage.get_local_path(raw_object_key, download_to=os.path.join(tmp_dir, "source"))
            width, height = _probe_video_dimensions(local_video_path)
        except Exception as exc:
            _mark_failed(clip_id, f"could not read source video: {exc}")
            log.error("rendering.permanent_failure", error=str(exc))
            return

        face_profile = None
        if settings.enable_face_aware_crop:
            try:
                sample_paths = _extract_face_sample_frames(local_video_path, start, end, tmp_dir)
                face_profile = estimate_face_profile(sample_paths)
            except Exception as exc:
                # Best-effort: any failure here (cv2 missing/broken, a bad
                # frame, whatever) just means "no face profile" -- falls
                # through to the original centered-crop behavior below,
                # never blocks or fails the render.
                log.warning("rendering.face_detect_failed", error=str(exc))
                face_profile = None

        # camera_layout_mode (StreamJob per-job override, see its model
        # docstring) short-circuits the usual face-size/corner-position
        # heuristic when the creator already knows the answer for this
        # VOD: "single_crop" means never split even if a face happens to
        # look reaction-shaped (a real false-positive a user hit -- see
        # README's "Reaction split layout" section), "split_reaction"
        # means always split when ANY face was found, skipping the
        # classifier's fixed thresholds entirely. Both still respect
        # settings.enable_reaction_split_layout as a hard global kill
        # switch -- a per-job "please split" override wouldn't make sense
        # to silently defeat an operator's decision to turn the whole
        # feature off. Either way, splitting still requires a face_profile
        # to exist -- there's no "cam" half to zoom in on if face
        # detection found nothing, so "split_reaction" degrades to
        # single_crop on a clip with no detected face, same as "auto"
        # already does.
        #
        # "fit_frame" is checked first and separately from the split/single
        # decision below, because it isn't a third *crop* -- it's the
        # absence of one (see build_fit_frame_filtergraph). It also
        # deliberately ignores enable_reaction_split_layout and face
        # detection entirely: there's nothing to classify when no cropping
        # happens, so neither can change the outcome.
        # The split layout's two panels, needed up front because a
        # hand-marked rect is resolved against its panel's aspect ratio
        # before the layout decision below, not inside it.
        #
        # Not halves any more: settings.split_facecam_fraction decides how
        # much height the facecam gets, and the content panel takes the rest.
        # Both must be even (yuv420p) and must sum to TARGET_HEIGHT exactly,
        # or vstack produces an output that is not 1080x1920.
        panel_w = TARGET_WIDTH
        cam_fraction = min(max(style['split_facecam_fraction'], 0.2), 0.8)
        cam_panel_h = int(TARGET_HEIGHT * cam_fraction) // 2 * 2
        main_panel_h = TARGET_HEIGHT - cam_panel_h  # even, since both TARGET_HEIGHT and cam_panel_h are
        cam_ratio = panel_w / cam_panel_h
        main_ratio = panel_w / main_panel_h

        # A facecam box the creator drew on a real frame of this VOD
        # (StreamJob.facecam_rect) outranks everything automatic: no Haar
        # detection to miss it, no classify_reaction_layout guess to get it
        # wrong. Detection answering "is there a facecam and where" was the
        # original source of the bad framing; a human pointing at it is the
        # one input that cannot be wrong.
        #
        # A malformed rect degrades to detection rather than failing the
        # render -- it is an optional hint, and a bad one should never cost
        # the clip.
        manual_cam_crop = None
        if facecam_rect:
            try:
                manual_cam_crop = crop_from_marked_rect(width, height, facecam_rect, cam_ratio)
            except ValueError as exc:
                log.warning("rendering.facecam_rect_invalid", error=str(exc), rect=facecam_rect)

        # The content region, marked the same way. Resolved against two
        # different target ratios because it feeds two different panels:
        # the split layout's bottom half, or -- when nothing is split -- the
        # whole 9:16 frame. A centered crop of a 16:9 gameplay frame throws
        # away two thirds of the width without knowing which third mattered;
        # this is how the creator says which third mattered.
        manual_main_crop = None
        manual_full_crop = None
        if gameplay_rect:
            try:
                manual_main_crop = crop_from_marked_rect(width, height, gameplay_rect, main_ratio)
                manual_full_crop = crop_from_marked_rect(
                    width, height, gameplay_rect, TARGET_WIDTH / TARGET_HEIGHT
                )
            except ValueError as exc:
                log.warning("rendering.gameplay_rect_invalid", error=str(exc), rect=gameplay_rect)
                manual_main_crop = None
                manual_full_crop = None

        is_fit_frame_layout = camera_layout_mode == "fit_frame"
        if is_fit_frame_layout:
            is_reaction_layout = False
        elif not settings.enable_reaction_split_layout:
            is_reaction_layout = False
        elif camera_layout_mode == "single_crop":
            # An explicit "no split" still wins over a marked rect: the
            # creator may have marked the facecam and then decided this VOD
            # reads better as one crop. The later, more specific instruction
            # is the one to honour.
            is_reaction_layout = False
        elif manual_cam_crop is not None:
            # Marked box + not explicitly told otherwise: there IS a
            # facecam, we know exactly where, so split. No face_profile
            # needed -- that was only ever a way of guessing this.
            is_reaction_layout = True
        elif camera_layout_mode == "split_reaction":
            is_reaction_layout = face_profile is not None
        else:  # None ("auto") -- the original heuristic, unchanged
            is_reaction_layout = (
                face_profile is not None
                and classify_reaction_layout(face_profile["center"], face_profile["area"], width, height)
            )
        log.info(
            "rendering.layout",
            layout=(
                "fit_frame" if is_fit_frame_layout
                else "split_reaction" if is_reaction_layout
                else "single_crop"
            ),
            face_found=face_profile is not None,
            camera_layout_mode=camera_layout_mode or "auto",
            crop_bias=crop_bias or "none",
            facecam_source=("marked" if manual_cam_crop is not None else "detected" if face_profile else "none"),
            gameplay_source=("marked" if manual_main_crop is not None else "auto"),
            split_panels=f"{cam_panel_h}/{main_panel_h}",
        )

        if is_fit_frame_layout:
            filtergraph = build_fit_frame_filtergraph(TARGET_WIDTH, TARGET_HEIGHT)
            # "Show the whole frame" used to mean literally the whole source
            # frame, which made it the one layout that ignored a marked
            # gameplay region -- and the only one that could have honoured a
            # WIDE one completely. A 16:9-ish region cannot fit inside a 9:16
            # crop (single_crop necessarily cuts its sides), but fit_frame
            # scales rather than crops, so the marked region survives intact.
            # With a mark, "whole frame" becomes "the whole of the part you
            # said matters" -- which is what someone who drew the box meant.
            if gameplay_rect:
                try:
                    src = marked_rect_pixels(width, height, gameplay_rect)
                except ValueError as exc:
                    log.warning("rendering.gameplay_rect_invalid", error=str(exc), rect=gameplay_rect)
                else:
                    crop_w, crop_h, crop_x, crop_y = src
                    filtergraph = (
                        f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y}[fitsrc];"
                        + retarget_source_label(filtergraph, "fitsrc")  # bare label: the helper adds the brackets
                    )
        elif is_reaction_layout:
            # Top: a tight zoom on the detected facecam. Bottom: the full
            # frame, plain-cropped to the same half-height target ratio --
            # see the module docstring for why this doesn't try to exclude
            # the webcam box's own footprint from the bottom half.
            if manual_cam_crop is not None:
                cam_crop = manual_cam_crop
            else:
                face_scale = face_profile["area"] ** 0.5
                cam_crop = compute_face_zoom_crop(width, height, face_profile["center"], face_scale, cam_ratio)
            if manual_main_crop is not None:
                main_w, main_h, main_x, main_y = manual_main_crop
            else:
                main_w, main_h = compute_vertical_crop(width, height, target_ratio=main_ratio)
                main_x, main_y = compute_crop_offset(
                    width, height, main_w, main_h, focal_point=None, bias=crop_bias
                )
            filtergraph = build_split_reaction_filtergraph(
                cam_crop, (main_w, main_h, main_x, main_y), panel_w, cam_panel_h, main_panel_h
            )
        else:
            if manual_full_crop is not None:
                # An explicitly marked content region beats both face
                # following and crop_bias: those exist to GUESS which part
                # of the frame matters, and this is being told.
                crop_w, crop_h, crop_x, crop_y = manual_full_crop
            else:
                crop_w, crop_h = compute_vertical_crop(width, height)
                focal_point = face_profile["center"] if face_profile else None
                crop_x, crop_y = compute_crop_offset(width, height, crop_w, crop_h, focal_point, bias=crop_bias)
            filtergraph = build_single_crop_filtergraph(crop_w, crop_h, crop_x, crop_y, TARGET_WIDTH, TARGET_HEIGHT)

        # Captions have to be remapped onto the stitched timeline for a
        # multi-part clip -- a caption from the second part plays at the
        # sum of the earlier parts' durations, not at its original source
        # timestamp (see build_multipart_srt).
        srt_content = (
            build_multipart_srt(transcript_segments, parts, max_chars=int(style['caption_max_chars']))
            if parts
            else build_srt(transcript_segments, start, end, max_chars=int(style['caption_max_chars']))
        )
        srt_path = None
        if srt_content:
            srt_path = os.path.join(tmp_dir, "captions.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

        # Title/hashtags/caption/explanation, one LLM call (heuristic
        # fallback if disabled/unavailable/failed) -- called here,
        # synchronously, rather than after rendering, specifically so the
        # title text is known in time to burn into the clip below. See the
        # module docstring for why this moved earlier.
        from app.core.llm_config import resolve_llm_config_for_user_id

        annotation = generate_caption_annotation(
            transcript_segments, start, end, score_breakdown, existing_caption,
            llm_config=resolve_llm_config_for_user_id(job_user_id),
        )
        log.info("rendering.annotation_generated", source=annotation["source"])

        title_srt_path = None
        if settings.enable_clip_title_overlay:
            title_content = build_title_srt(annotation["title"], clip_duration)
            if title_content:
                title_srt_path = os.path.join(tmp_dir, "title.srt")
                with open(title_srt_path, "w", encoding="utf-8") as f:
                    f.write(title_content)

        out_path = os.path.join(tmp_dir, "clip.mp4")
        flag_reasons: list[str] = []
        try:
            _render(
                local_video_path, out_path, start, end, filtergraph,
                srt_path, title_srt_path, style, parts=parts,
            )
        except RuntimeError as exc:
            if srt_path is None and title_srt_path is None:
                _mark_failed(clip_id, f"ffmpeg render failed: {exc}")
                log.error("rendering.permanent_failure", error=str(exc))
                return
            # Text burn-in (libass) is the step most likely to behave
            # differently across ffmpeg builds -- degrade to a plain render
            # (no captions, no title) rather than losing the whole clip
            # over it. Flag whichever stage(s) were actually dropped so a
            # reviewer sees why a clip is missing text it should have had.
            log.warning("rendering.burned_text_failed_retrying_without", error=str(exc))
            if srt_path is not None:
                flag_reasons.append("captions_failed")
            if title_srt_path is not None:
                flag_reasons.append("title_failed")
            try:
                _render(
                    local_video_path, out_path, start, end, filtergraph,
                    None, None, style, parts=parts,
                )
            except RuntimeError as exc2:
                _mark_failed(clip_id, f"ffmpeg render failed (with and without burned-in text): {exc2}")
                log.error("rendering.permanent_failure", error=str(exc2))
                return

        thumb_path = os.path.join(tmp_dir, "thumb.jpg")
        try:
            _extract_thumbnail(out_path, thumb_path, clip_duration, tmp_dir)
        except RuntimeError as exc:
            # A missing thumbnail is a reviewer-experience issue, not a
            # reason to throw away an otherwise-good clip.
            log.warning("rendering.thumbnail_failed", error=str(exc))
            thumb_path = None

        video_key = new_object_key(f"clips/{stream_job_id}", "mp4")
        storage.put_file(out_path, video_key)

        thumb_key = None
        if thumb_path and os.path.exists(thumb_path):
            thumb_key = new_object_key(f"clips/{stream_job_id}", "jpg")
            storage.put_file(thumb_path, thumb_key)

        with db_session() as db:
            clip = db.get(RenderedClip, clip_id)
            candidate = db.get(CandidateSegment, candidate_id)
            if clip is None:
                # Deleted mid-render (e.g. a reviewer deleted the stream_job)
                # -- the video/thumbnail were already uploaded to storage
                # above, orphaned but harmless; nothing left to update.
                log.warning("rendering.clip_row_gone_before_finish")
                return
            clip.status = "rendered"
            clip.object_key = video_key
            clip.thumbnail_key = thumb_key
            clip.duration_seconds = clip_duration
            clip.flag_reasons = flag_reasons
            clip.last_error = None
            clip.caption_text = annotation["caption"]
            if candidate is not None:
                candidate.llm_annotation = annotation

        log.info(
            "rendering.done",
            video_key=video_key,
            thumbnail_key=thumb_key,
            flag_reasons=flag_reasons,
            annotation_source=annotation["source"],
        )

    finally:
        for name in ("source", "captions.srt", "title.srt", "clip.mp4", "thumb.jpg"):
            path = os.path.join(tmp_dir, name)
            if os.path.exists(path):
                os.remove(path)
        for i in range(len(_FACE_SAMPLE_FRACTIONS)):
            path = os.path.join(tmp_dir, f"face_sample_{i}.jpg")
            if os.path.exists(path):
                os.remove(path)
        for i in range(len(_THUMBNAIL_SAMPLE_FRACTIONS)):
            path = os.path.join(tmp_dir, f"thumb_candidate_{i}.jpg")
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


def _mark_failed(clip_id: uuid.UUID, error: str) -> None:
    with db_session() as db:
        clip = db.get(RenderedClip, clip_id)
        if clip:
            clip.status = "failed"
            clip.last_error = error[:4000]


def on_failure(job, connection, type, value, traceback) -> None:
    """RQ failure callback -- fires once retries are exhausted (see app.core.queue.enqueue)."""
    candidate_segment_id = job.args[0] if job.args else None
    with db_session() as db:
        clip = (
            db.query(RenderedClip)
            .filter(RenderedClip.candidate_segment_id == uuid.UUID(candidate_segment_id))
            .order_by(RenderedClip.created_at.desc())
            .first()
            if candidate_segment_id
            else None
        )
        if clip:
            clip.status = "failed"
            clip.last_error = f"{type.__name__}: {value}"[:4000]
    logger.bind(candidate_segment_id=candidate_segment_id, worker="rendering").error(
        "rendering.retries_exhausted", error=str(value)
    )
