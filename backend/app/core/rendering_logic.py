"""Pure helpers for the rendering worker -- no DB/IO/subprocess calls, so
these are unit-testable without real ffmpeg (see tests/test_rendering_logic.py).
Same split as scoring_logic.py/segmentation_logic.py: keep the parts that
are just math/string-building out of the worker so they're fast to test and
easy to reason about independent of ffmpeg actually running.
"""
from __future__ import annotations

TARGET_ASPECT_RATIO = 9 / 16  # vertical short-form -- see architecture doc §8


def _even(n: int) -> int:
    """Round down to the nearest even number, floor at 2.

    libx264 + yuv420p require even width/height -- an odd crop dimension
    fails the encode outright, not a cosmetic issue.
    """
    n = int(n)
    return max(2, n - (n % 2))


def compute_vertical_crop(width: int, height: int, target_ratio: float = TARGET_ASPECT_RATIO) -> tuple[int, int]:
    """Centered crop dimensions (crop_w, crop_h) that achieve target_ratio
    from a `width`x`height` source, cropping the long axis only.

    A fixed center-crop, not motion-tracked/smart reframing -- that's an
    explicit MVP trade-off (architecture doc §8: face/subject tracking is
    real complexity for a signal we don't have yet). ffmpeg's crop filter
    centers automatically when x/y are omitted, so the caller just needs
    these dimensions.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    if (width / height) > target_ratio:
        # Source is wider than the target -- crop width, keep full height.
        crop_h = _even(height)
        crop_w = _even(height * target_ratio)
    else:
        # Source is already narrower/taller than the target (or exactly at
        # it) -- crop height, keep full width.
        crop_w = _even(width)
        crop_h = _even(width / target_ratio)

    return crop_w, crop_h


def compute_crop_offset(
    width: int,
    height: int,
    crop_w: int,
    crop_h: int,
    focal_point: tuple[float, float] | None = None,
    bias: str | None = None,
) -> tuple[int, int]:
    """Top-left (x, y) for ffmpeg's `crop` filter, given dimensions already
    computed by compute_vertical_crop.

    With no focal point (the original behavior), centers the crop on the
    source frame. With one -- e.g. a detected face center from
    app.core.face_detect -- centers the crop on that point instead, clamped
    so the window never runs off either edge. Only the axis that's actually
    being cropped can move; the other axis already uses the source's full
    extent (crop_w == width or crop_h == height), so there's nothing to
    offset there regardless of the focal point.

    `bias` ("left"/"center"/"right", from StreamJob.crop_bias) is an
    explicit per-job override and takes precedence over `focal_point`
    entirely -- same principle as camera_layout_mode overriding
    classify_reaction_layout: when a creator has told us where the
    important part of their frame is, an automatic guess should not get to
    argue. "left"/"right" pin the crop flush to that edge; "center" is the
    original centered behavior with face-following explicitly disabled.
    Only meaningful on the axis actually being cropped (horizontal for the
    usual landscape source); a bias on an axis with nothing to crop is a
    no-op rather than an error.
    """
    if bias in ("left", "center", "right"):
        if bias == "left":
            x = 0
        elif bias == "right":
            x = width - crop_w
        else:
            x = (width - crop_w) // 2
        return max(0, min(x, width - crop_w)), (height - crop_h) // 2

    if focal_point is None:
        return (width - crop_w) // 2, (height - crop_h) // 2

    focal_x, focal_y = focal_point
    x = int(focal_x - crop_w / 2) if crop_w < width else 0
    y = int(focal_y - crop_h / 2) if crop_h < height else 0
    x = max(0, min(x, width - crop_w))
    y = max(0, min(y, height - crop_h))
    return x, y


# --- TikTok UI safe zones, 1080x1920 vertical (researched 2026-09-11) ---
#
# TikTok overlays its own chrome on top of every video in the For You feed.
# Anything rendered underneath it is still "in the clip" but is not visible
# to a viewer. Published figures vary between sources (they are measured by
# creators against the live app, not documented by TikTok), so these take
# the more conservative end of the range found:
#   top    ~200px (10.4%) -- search icon, LIVE badge, following/for-you tabs
#   bottom ~334px (17.4%) -- @username, caption text, audio marquee (organic;
#                            ads reserve ~450px for the CTA button instead)
#   right  ~140px (13.0%) -- avatar, like, comment, bookmark, share rail
#   left    ~86px  (8.0%) -- bezel/rounded-corner buffer
# Treat these as "known to within a few percent," not gospel -- re-measure
# against the real app before spending much on the last 20px either way.
TIKTOK_SAFE_TOP_PX = 200
TIKTOK_SAFE_BOTTOM_PX = 334
TIKTOK_SAFE_RIGHT_PX = 140
TIKTOK_SAFE_LEFT_PX = 86

# Fraction of the total frame area a detected face can occupy and still
# count as a small "webcam box" rather than someone filling the frame --
# see classify_reaction_layout.
#
# 0.18 -> 0.06 (2026-09-11), after the user reported the framing "not
# showing the most important parts" and split layouts firing on clips with
# no facecam. 0.18 of a 1920x1080 frame is a 611x611px face box -- far
# larger than any real webcam overlay (a typical one is 120-250px across,
# i.e. 1-2% of frame area), so the "is it small?" half of this test was
# passing for essentially every face including full IRL talking heads.
# 0.06 (a ~353x353px box) still comfortably admits a real webcam face
# while excluding someone who is themselves the subject.
_REACTION_MAX_FACE_AREA_FRACTION = 0.06
# How close to a corner (as a fraction of width/height) a face's center has
# to be on BOTH axes to count as "in a corner".
#
# 0.4 -> 0.25 (2026-09-11), same root cause. At 0.4 the outer 40% on each
# axis counted as "a corner", leaving only the central 20%x20% excluded --
# so 96% of all possible face positions qualified, and the test was very
# nearly "is the face not dead-center?". Measured against realistic
# scenarios, that misclassified 4 of 8, including every rule-of-thirds IRL
# framing (a person at x=33%,y=35% is normal composition, not a webcam
# box). At 0.25 the central 50%x50% is excluded, which is what "in a
# corner" was always meant to mean.
_REACTION_CORNER_MARGIN_FRACTION = 0.25


def classify_reaction_layout(
    face_center: tuple[float, float],
    face_area: float,
    frame_width: int,
    frame_height: int,
    *,
    max_area_fraction: float = _REACTION_MAX_FACE_AREA_FRACTION,
    corner_margin_fraction: float = _REACTION_CORNER_MARGIN_FRACTION,
) -> bool:
    """True if a detected face looks like a small, corner-positioned webcam
    box layered over separate main content (a "reacting" clip -- streamer
    reacting to gameplay/a video elsewhere in frame) rather than one large,
    roughly-centered face filling most of the frame (an "IRL" clip -- a
    single handheld/selfie camera). Two conditions, both required: small
    (at most `max_area_fraction` of the total frame) AND within
    `corner_margin_fraction` of the frame's own width/height from some
    corner on both axes -- a small face floating near dead-center doesn't
    count either, since that's more likely just someone standing far from
    an IRL camera than a webcam overlay.

    A heuristic, not a learned classifier -- same "cheap deterministic
    signal, good enough" approach as app.core.scoring_logic. A false
    negative here (a real reaction layout not detected as one) just means
    that clip gets the plain single face-aware crop instead of the nicer
    split treatment -- never a worse or broken render, see
    app.workers.rendering for how this gates the layout choice.
    """
    frame_area = frame_width * frame_height
    if frame_area <= 0:
        return False
    if (face_area / frame_area) > max_area_fraction:
        return False

    face_x, face_y = face_center
    near_left = face_x <= frame_width * corner_margin_fraction
    near_right = face_x >= frame_width * (1 - corner_margin_fraction)
    near_top = face_y <= frame_height * corner_margin_fraction
    near_bottom = face_y >= frame_height * (1 - corner_margin_fraction)
    return (near_left or near_right) and (near_top or near_bottom)


def marked_rect_pixels(width: int, height: int, rect: dict) -> tuple[int, int, int, int]:
    """(w, h, x, y) for a marked region, at its OWN aspect ratio.

    Unlike crop_from_marked_rect, this does not grow the box towards a target
    ratio -- it is for the fit-frame layout, which scales whatever it is given
    to fit the output width over a blurred background and therefore does not
    need a particular input shape. Keeping the region's own proportions is the
    whole point there: it is the one layout that can show a wide gameplay
    region in full, with nothing cropped away.

    Raises ValueError on a malformed rect, same contract as
    crop_from_marked_rect, so callers can fall back to the whole frame.
    """
    try:
        x = float(rect["x"]) * width
        y = float(rect["y"]) * height
        w = float(rect["w"]) * width
        h = float(rect["h"]) * height
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed marked rect: {rect!r}") from exc

    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"marked rect has no area: {rect!r}")

    def _even_down(value: float) -> int:
        return max(0, int(value) // 2 * 2)

    def _even_up(value: float) -> int:
        rounded = int(value)
        if rounded < value:
            rounded += 1
        return max(2, rounded + (rounded % 2))

    out_w = min(_even_up(w), _even_down(width))
    out_h = min(_even_up(h), _even_down(height))
    out_x = _even_down(min(max(0.0, x), width - out_w))
    out_y = _even_down(min(max(0.0, y), height - out_h))
    return out_w, out_h, out_x, out_y


def crop_from_marked_rect(
    width: int,
    height: int,
    rect: dict,
    target_ratio: float,
) -> tuple[int, int, int, int]:
    """(crop_w, crop_h, crop_x, crop_y) for a hand-marked facecam box.

    `rect` is a marked region -- StreamJob.facecam_rect or
    StreamJob.gameplay_rect: {"x","y","w","h"} as fractions of the source
    frame (0..1), drawn by the creator on a real frame of their own VOD.
    `target_ratio` is the width/height of the panel it has to fill (the
    split layout's top or bottom half, or the full 9:16 frame for a single
    crop).

    The marked box is treated as the MINIMUM that must stay visible: the
    window is grown (never shrunk) on whichever axis is short of
    `target_ratio`, centered on the box, so the creator always gets at
    least what they drew. If the resulting window would leave the frame it
    is clamped inside it, and if it cannot fit at all the largest window
    with the right aspect ratio is used instead -- shrinking rather than
    returning an off-frame or wrong-ratio crop, since ffmpeg would either
    error or stretch the picture.

    Unlike compute_face_zoom_crop (which pads out from a detected face
    center by a guessed multiple of the face size) there is no guesswork
    here -- the creator drew the box.

    Raises ValueError on a malformed rect so the caller can fall back to
    detection rather than rendering a garbage crop; callers should treat
    that as "behave as if no rect was set".
    """
    try:
        rx = float(rect["x"]) * width
        ry = float(rect["y"]) * height
        rw = float(rect["w"]) * width
        rh = float(rect["h"]) * height
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed facecam_rect: {rect!r}") from exc

    if rw <= 0 or rh <= 0 or width <= 0 or height <= 0 or target_ratio <= 0:
        raise ValueError(f"facecam_rect has no area: {rect!r}")

    center_x = rx + rw / 2.0
    center_y = ry + rh / 2.0

    # Grow the deficient axis so the box is never cropped tighter than drawn.
    if rw / rh < target_ratio:
        crop_h = rh
        crop_w = rh * target_ratio
    else:
        crop_w = rw
        crop_h = rw / target_ratio

    # Cannot exceed the source frame; shrink preserving the ratio if it does.
    if crop_w > width:
        crop_w = float(width)
        crop_h = crop_w / target_ratio
    if crop_h > height:
        crop_h = float(height)
        crop_w = crop_h * target_ratio

    crop_x = center_x - crop_w / 2.0
    crop_y = center_y - crop_h / 2.0
    crop_x = max(0.0, min(crop_x, width - crop_w))
    crop_y = max(0.0, min(crop_y, height - crop_h))

    # Even numbers everywhere: yuv420p chroma subsampling needs even
    # dimensions, and an odd crop makes ffmpeg fail or silently shift a
    # pixel. Sizes round UP and offsets round DOWN, deliberately and in
    # opposite directions -- rounding both the same way can shave a pixel
    # or two off the box the creator actually drew, which is the one thing
    # this function exists to preserve. Offsets floor to 0, not to 2: a
    # crop starting at the very edge of the frame is completely normal
    # (a facecam pinned to a corner), and forcing it to 2 would nudge the
    # window off the mark for exactly the most common case.
    def _even_down(value: float) -> int:
        return max(0, int(value) // 2 * 2)

    def _even_up(value: float) -> int:
        rounded = int(value)
        if rounded < value:
            rounded += 1
        return max(2, rounded + (rounded % 2))

    out_w = min(_even_up(crop_w), _even_down(width))
    out_h = min(_even_up(crop_h), _even_down(height))
    out_x = _even_down(min(crop_x, width - out_w))
    out_y = _even_down(min(crop_y, height - out_h))
    return out_w, out_h, out_x, out_y


def compute_face_zoom_crop(
    width: int,
    height: int,
    face_center: tuple[float, float],
    face_scale: float,
    target_ratio: float,
    padding: float = 3.2,
) -> tuple[int, int, int, int]:
    """(crop_w, crop_h, x, y) for a `target_ratio` crop that gives a
    detected face comfortable headroom (`padding` x its own approximate
    size), centered on it and clamped to the source frame -- the "webcam"
    half of a split reaction-layout render (see app.workers.rendering and
    classify_reaction_layout above). `face_scale` is the face's
    approximate bounding-box side length (e.g. sqrt of its detected pixel
    area -- Haar-cascade face boxes are close enough to square that this is
    a reasonable single-number size estimate without needing separate
    width/height from the detector).
    """
    desired_h = max(face_scale * padding, 20.0)
    desired_w = desired_h * target_ratio
    if desired_w > width:
        desired_w = float(width)
        desired_h = desired_w / target_ratio
    if desired_h > height:
        desired_h = float(height)
        desired_w = desired_h * target_ratio

    crop_w = min(_even(desired_w), _even(width))
    crop_h = min(_even(desired_h), _even(height))
    x, y = compute_crop_offset(width, height, crop_w, crop_h, focal_point=face_center)
    return crop_w, crop_h, x, y


def build_single_crop_filtergraph(crop_w: int, crop_h: int, x: int, y: int, target_w: int, target_h: int) -> str:
    """ffmpeg `-filter_complex` graph for the plain (non-split) crop+scale
    path -- always ends in the `[vout]` label callers (see _render in
    app.workers.rendering) map to the output file, same convention as
    build_split_reaction_filtergraph below so callers don't need to know
    which layout produced the graph.
    """
    return f"[0:v]crop={crop_w}:{crop_h}:{x}:{y},scale={target_w}:{target_h},setsar=1[vout]"


# Blur strength for the fit-frame layout's background fill. Strong enough
# that the background reads as texture rather than a competing second copy
# of the video (a lightly-blurred duplicate is visually noisy and pulls the
# eye away from the real content band), cheap enough not to matter next to
# the libx264 encode already happening in the same pass.
_FIT_FRAME_BLUR_RADIUS = 40
_FIT_FRAME_BLUR_POWER = 2


def build_fit_frame_filtergraph(target_w: int, target_h: int) -> str:
    """ffmpeg `-filter_complex` graph for the "show the whole frame"
    layout: the ENTIRE source frame scaled to the output width and
    centered vertically, over a blurred, zoomed copy of itself filling the
    leftover space above and below.

    Exists because the other two layouts both *crop*, and cropping a 16:9
    source to 9:16 throws away roughly two thirds of the width -- fine when
    the subject is a person who can be centered, actively harmful for
    gameplay/screen-share content where the important thing (a scoreboard,
    a UI element, the other half of the map) is exactly what falls outside
    the crop window. This layout loses nothing; the trade-off is that the
    content band occupies less of the screen vertically, which is the right
    trade whenever "all of it, smaller" beats "some of it, bigger".

    `scale={target_w}:-2` keeps the source's own aspect ratio and forces an
    even height (libx264/yuv420p reject odd dimensions, same constraint
    _even() handles for the crop paths). The background branch scales with
    `force_original_aspect_ratio=increase` then crops, so it always fully
    covers the canvas no matter the source's shape -- no black bars can
    show through at the edges. Ends in `[vout]` like every other builder
    here so append_subtitles_stage and _render treat it identically.
    """
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},"
        f"boxblur=luma_radius={_FIT_FRAME_BLUR_RADIUS}:luma_power={_FIT_FRAME_BLUR_POWER}[bgblur];"
        f"[fg]scale={target_w}:-2,setsar=1[fgscaled];"
        f"[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2,setsar=1[vout]"
    )


def build_split_reaction_filtergraph(
    cam_crop: tuple[int, int, int, int],
    main_crop: tuple[int, int, int, int],
    out_w: int,
    cam_out_h: int,
    main_out_h: int | None = None,
) -> str:
    """ffmpeg `-filter_complex` graph for the top-facecam/bottom-content
    split layout: two independent crop+scale branches off the same source
    frame (`[0:v]`), stacked vertically with ffmpeg's `vstack` filter.

    Both branches must scale to the same WIDTH -- vstack rejects mismatched
    widths outright. Their heights are independent, which is the point: a
    50/50 split gives the webcam as much of the screen as the gameplay,
    which is rarely what the clip is about. `main_out_h` defaults to
    `cam_out_h` for the old equal-halves behaviour.

    The caller is responsible for the two heights summing to the output
    height and both being even (yuv420p chroma subsampling).
    """
    if main_out_h is None:
        main_out_h = cam_out_h
    cam_w, cam_h, cam_x, cam_y = cam_crop
    main_w, main_h, main_x, main_y = main_crop
    return (
        f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},scale={out_w}:{cam_out_h},setsar=1[cam];"
        f"[0:v]crop={main_w}:{main_h}:{main_x}:{main_y},scale={out_w}:{main_out_h},setsar=1[main];"
        f"[cam][main]vstack=inputs=2[vout]"
    )


def append_subtitles_stage(filtergraph: str, srt_path: str, style: str) -> str:
    """Chains a subtitles burn-in stage onto a `[vout]`-terminated
    filter_complex graph (either builder above) -- relabels the existing
    output to an intermediate name and adds the subtitles filter as the new
    final `[vout]`, so the caller doesn't need to know or care whether the
    graph came from the single-crop or split-reaction path.
    """
    escaped = ffmpeg_subtitles_filter_path(srt_path)
    relabeled = filtergraph.replace("[vout]", "[vpre]", 1)
    return f"{relabeled};[vpre]subtitles='{escaped}':force_style='{style}'[vout]"


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# Two lines at the default caption font size (~44 characters per line,
# measured against a real libass render on a 1080x1920 frame). Callers that
# have the app config pass settings.caption_max_chars instead; keeping the
# default here is what lets this module stay free of a config import, which
# is what makes it unit-testable with no app wiring.
DEFAULT_CAPTION_MAX_CHARS = 80


def split_caption_text(text: str, max_chars: int) -> list[str]:
    """Break one transcript segment into caption-sized pieces.

    Whisper emits a whole sentence per segment, so burning segments in
    verbatim produced four-line paragraphs covering a third of the frame
    (measured: 256px tall, 13.3% of a 1080x1920 clip). Short-form captions
    want one or two short lines, replaced often.

    Splits at the latest sentence boundary that fits, else the latest clause
    boundary, else the latest space -- so a break lands where a reader would
    pause rather than mid-thought. A single word longer than max_chars is
    emitted whole rather than chopped: one slightly over-wide line beats an
    unreadable fragment.
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        cut = -1
        for group in ((". ", "! ", "? "), (", ", "; ", " -- "), (" ",)):
            for mark in group:
                found = window.rfind(mark)
                if found > 0:
                    # keep sentence/clause punctuation with the chunk it ends
                    candidate = found + (len(mark) if mark != " " else 0)
                    cut = max(cut, candidate)
            if cut > 0:
                break
        if cut <= 0:
            space = remaining.find(" ")
            cut = space if space > 0 else len(remaining)
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


def _timed_caption_chunks(
    rel_start: float, rel_end: float, text: str, max_chars: int
) -> list[tuple[float, float, str]]:
    """Split `text` and share the segment's own duration between the pieces,
    proportional to length -- a chunk twice as long stays up twice as long,
    a decent proxy for how long it takes to say."""
    chunks = split_caption_text(text, max_chars)
    if len(chunks) <= 1:
        return [(rel_start, rel_end, chunks[0])] if chunks else []

    total_chars = sum(len(c) for c in chunks)
    span = max(rel_end - rel_start, 0.001)
    out: list[tuple[float, float, str]] = []
    cursor = rel_start
    for i, chunk in enumerate(chunks):
        # The last chunk takes whatever is left, so rounding can never leave a
        # gap or push the final caption past the segment's end.
        chunk_end = rel_end if i == len(chunks) - 1 else min(cursor + span * (len(chunk) / total_chars), rel_end)
        out.append((cursor, chunk_end, chunk))
        cursor = chunk_end
    return out


def build_srt(
    transcript_segments: list[dict],
    start: float,
    end: float,
    max_chars: int = DEFAULT_CAPTION_MAX_CHARS,
) -> str:
    """SRT subtitle content for the transcript slice covering [start, end],
    with timestamps shifted so 0 == the clip's own start. Returns "" if
    nothing in the window has text -- callers should skip the subtitles
    filter entirely in that case rather than burning in an empty track.

    One subtitle entry per transcript segment (clipped to the window
    boundary) -- a deliberate v1 simplification, same spirit as
    segmentation's "simple first" choices. Per-word karaoke-style
    highlighting is a nicer short-form caption style but real added
    complexity for a signal (word-level timestamps) not every STT provider
    reliably returns; flagged as a later enhancement, not MVP scope.
    """
    entries = []
    for seg in sorted(transcript_segments, key=lambda s: s.get("start", 0)):
        seg_start = max(seg.get("start", 0.0), start)
        seg_end = min(seg.get("end", 0.0), end)
        if seg_end <= seg_start:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        entries.append((seg_start - start, seg_end - start, text))

    if not entries:
        return ""

    blocks = []
    idx = 0
    for rel_start, rel_end, text in entries:
        for chunk_start, chunk_end, chunk in _timed_caption_chunks(
            rel_start, rel_end, text, max_chars
        ):
            idx += 1
            blocks.append(
                f"{idx}\n{_format_srt_timestamp(chunk_start)} --> "
                f"{_format_srt_timestamp(chunk_end)}\n{chunk}\n"
            )
    return "\n".join(blocks)


def normalize_parts(raw_parts) -> list[tuple[float, float]] | None:
    """Coerce a stored `candidate_segments.parts` value (JSONB -- a list of
    `[start, end]` pairs) into a validated list of float tuples, or None if
    it doesn't describe a usable multi-part clip.

    None means "treat this as an ordinary single-window candidate," which
    is what every candidate before this feature existed still is: parts is
    nullable and the whole multi-part path is skipped when it's absent.
    Returning None rather than raising on malformed input is deliberate --
    a bad parts value should degrade a clip to its plain [start, end]
    window, never fail the render outright.

    A single part is also treated as "not multi-part": there is nothing to
    stitch, and going through the concat filtergraph for one segment would
    add an encode stage for no benefit.
    """
    if not isinstance(raw_parts, (list, tuple)) or len(raw_parts) < 2:
        return None

    parts: list[tuple[float, float]] = []
    for item in raw_parts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            part_start, part_end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
        if part_end <= part_start or part_start < 0:
            return None
        parts.append((part_start, part_end))

    parts.sort(key=lambda p: p[0])
    # Overlapping parts would play the same footage twice with a visible
    # jump between the copies -- almost certainly a proposer bug rather
    # than an intentional edit, so fall back to the single-window path.
    for previous, following in zip(parts, parts[1:]):
        if following[0] < previous[1]:
            return None
    return parts


def parts_duration(parts: list[tuple[float, float]]) -> float:
    """Total playing time of a stitched clip -- the sum of its parts, NOT
    the span from first start to last end (the gaps between parts are
    exactly what gets cut out)."""
    return sum(end - start for start, end in parts)


def build_concat_prefix(parts: list[tuple[float, float]], include_audio: bool) -> str:
    """Filtergraph stages that cut each part out of the source and join
    them into one continuous stream, ending in `[vsrc]` (and `[asrc]` when
    `include_audio`).

    This is a PREFIX, not a complete graph: the layout builders above all
    read from `[0:v]`, so a caller stitches by prepending this and then
    calling retarget_source_label(layout_graph, "vsrc") so the layout
    operates on the concatenated stream instead of the raw input. That
    keeps every layout (single crop, split reaction, fit frame) working
    with multi-part clips without any of them knowing multi-part exists.

    `setpts=PTS-STARTPTS` (and its audio twin `asetpts`) on each part is
    what makes this work: `trim` keeps the source's original timestamps,
    so without rebasing each part to zero, `concat` would produce a stream
    with huge gaps matching the material that was cut out.

    Audio is opt-in per call because the source may genuinely have no audio
    track -- the existing single-window path handles that with ffmpeg's
    `-map 0:a?` (optional mapping), but a filtergraph referencing a
    nonexistent `[0:a]` fails outright rather than degrading, so the caller
    has to probe first (see app.workers.rendering._probe_has_audio).
    """
    stages: list[str] = []
    for i, (part_start, part_end) in enumerate(parts):
        stages.append(
            f"[0:v]trim=start={part_start:.3f}:end={part_end:.3f},setpts=PTS-STARTPTS[cv{i}]"
        )
    video_inputs = "".join(f"[cv{i}]" for i in range(len(parts)))
    stages.append(f"{video_inputs}concat=n={len(parts)}:v=1:a=0[vsrc]")

    if include_audio:
        for i, (part_start, part_end) in enumerate(parts):
            stages.append(
                f"[0:a]atrim=start={part_start:.3f}:end={part_end:.3f},asetpts=PTS-STARTPTS[ca{i}]"
            )
        audio_inputs = "".join(f"[ca{i}]" for i in range(len(parts)))
        stages.append(f"{audio_inputs}concat=n={len(parts)}:v=0:a=1[asrc]")

    return ";".join(stages)


def retarget_source_label(filtergraph: str, new_label: str) -> str:
    """Point a layout graph at an intermediate stream instead of the raw
    input, by rewriting every `[0:v]` reference to `[new_label]`.

    Every builder in this module reads from `[0:v]`, including the ones
    that read it more than once (build_split_reaction_filtergraph's two
    crop branches, build_fit_frame_filtergraph's split), so this replaces
    all occurrences rather than just the first -- the mirror image of
    append_subtitles_stage, which rewrites only the FIRST `[vout]` because
    there is exactly one final output label.
    """
    return filtergraph.replace("[0:v]", f"[{new_label}]")


def build_multipart_srt(
    transcript_segments: list[dict],
    parts: list[tuple[float, float]],
    max_chars: int = DEFAULT_CAPTION_MAX_CHARS,
) -> str:
    """SRT content for a stitched clip: each part's transcript slice
    shifted onto the stitched timeline, where part N starts at the summed
    duration of all parts before it.

    build_srt's single-window version can't be reused directly here
    precisely because of that remapping -- a caption from the second part
    has to be moved back by however much source material was cut out
    between parts, which depends on every earlier part's length, not just
    the clip's own start.
    """
    entries: list[tuple[float, float, str]] = []
    elapsed = 0.0
    for part_start, part_end in parts:
        for seg in sorted(transcript_segments, key=lambda s: s.get("start", 0)):
            seg_start = max(seg.get("start", 0.0), part_start)
            seg_end = min(seg.get("end", 0.0), part_end)
            if seg_end <= seg_start:
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            entries.append((elapsed + (seg_start - part_start), elapsed + (seg_end - part_start), text))
        elapsed += part_end - part_start

    if not entries:
        return ""

    blocks = []
    idx = 0
    for rel_start, rel_end, text in entries:
        for chunk_start, chunk_end, chunk in _timed_caption_chunks(
            rel_start, rel_end, text, max_chars
        ):
            idx += 1
            blocks.append(
                f"{idx}\n{_format_srt_timestamp(chunk_start)} --> "
                f"{_format_srt_timestamp(chunk_end)}\n{chunk}\n"
            )
    return "\n".join(blocks)


def build_title_srt(title: str, duration: float) -> str:
    """SRT content for a single cue spanning the whole clip -- the burned-in
    clickbait-style title banner shown at the top of the frame (see
    app.workers.rendering._TITLE_STYLE for the actual positioning/font;
    this file has no positioning of its own, chained as a second
    `subtitles` burn-in stage via append_subtitles_stage, same separation
    of concerns as the transcript captions' build_srt above -- content here,
    style/position at the call site). Returns "" for a blank/whitespace-only
    title so callers can skip the extra subtitles stage entirely rather
    than burning in an empty banner.
    """
    title = (title or "").strip()
    if not title:
        return ""
    return f"1\n{_format_srt_timestamp(0.0)} --> {_format_srt_timestamp(max(duration, 0.01))}\n{title}\n"


def ffmpeg_subtitles_filter_path(path: str) -> str:
    """Escape a local filesystem path for use as the `subtitles` ffmpeg
    filter's file argument.

    ffmpeg filtergraph syntax treats ':' as a key=value separator, which
    collides with Windows drive-letter paths (e.g. "C:\\Users\\...") -- a
    known, well-documented ffmpeg gotcha, not specific to this project. The
    project runs on the user's Windows machine (see project notes on other
    Windows-only surprises), so this is handled proactively rather than
    waiting for it to surface as a live bug report: normalize to forward
    slashes and escape any colon.
    """
    return path.replace("\\", "/").replace(":", "\\:")
