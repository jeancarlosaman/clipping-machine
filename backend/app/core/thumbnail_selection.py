"""Pick a good representative frame for a rendered clip's thumbnail,
instead of always grabbing whatever frame happens to sit at a fixed offset
into the clip (the original behavior -- see app.workers.rendering's
`_extract_thumbnail`, before this module existed).

Same "cheap, deterministic feature scoring, no new dependency" posture as
app.core.scoring_logic and app.core.face_detect -- this reuses
app.core.face_detect.detect_largest_face (already-loaded OpenCV Haar
cascade, no model download) plus two more OpenCV/numpy one-liners
(grayscale mean for brightness, Laplacian variance for sharpness). No
network call, no ML training, and no claim that this finds THE best frame
in any generative sense -- it's a ranking over a handful of sampled
candidates from the clip itself, same spirit as
app.core.scoring_logic.select_top_non_overlapping picking among candidate
windows rather than generating new ones.

Deliberately scores candidates *relative to each other* (min-max normalized
sharpness within one clip's own candidate set) rather than against a
hand-picked global sharpness cap -- unlike scoring_logic.py's
_NORMALIZATION_CAPS (which need to be comparable *across* jobs/clips for a
reviewer's displayed score to mean something stable), thumbnail selection
only ever needs to answer "which of these frames from this one clip looks
best," so there is nothing to mis-calibrate a global cap against and no
reason to guess one.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.face_detect import detect_largest_face
from app.workers.common import logger

# Below this mean grayscale value (0..255), treat a frame as "dead" -- most
# likely a fade-to-black transition or a genuinely black scene, not a
# usable thumbnail. Above this, likewise for fade-to-white / blown-out
# frames. Deliberately generous (not "reject anything slightly dark") --
# streaming content is often dim, and this should only catch the extreme,
# unambiguous case of a transition frame.
_DARK_BRIGHTNESS_THRESHOLD = 15.0
_BRIGHT_BRIGHTNESS_THRESHOLD = 240.0

# How much weight normalized sharpness vs. "a face is visible" get in the
# composite. Face presence gets the larger, flat share deliberately: a
# recognizable face is well-established creator advice for thumbnails that
# get clicked, and a sharp-but-faceless frame (e.g. a static gameplay HUD)
# is a weaker default than a merely-decent frame that shows the streamer.
# Both are guesses -- there's no click-through data in this project to
# calibrate against yet, same caveat as every SCORE_WEIGHT_* in
# scoring_logic.py.
_SHARPNESS_WEIGHT = 0.5
_FACE_PRESENCE_BONUS = 0.5


@dataclass(frozen=True)
class FrameCandidate:
    path: str
    brightness: float
    sharpness: float
    has_face: bool
    is_dead_frame: bool
    composite: float = 0.0


def _analyze_frame(path: str) -> FrameCandidate | None:
    """Reads one frame image and computes its raw signals. Returns None if
    the frame can't be read at all (corrupt/missing) -- callers drop it
    from the candidate pool rather than crashing thumbnail selection over
    one bad extract."""
    import cv2

    image = cv2.imread(path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_dead_frame = brightness < _DARK_BRIGHTNESS_THRESHOLD or brightness > _BRIGHT_BRIGHTNESS_THRESHOLD

    has_face = False
    try:
        has_face = detect_largest_face(path) is not None
    except Exception as exc:  # defensive, same posture as face_detect.estimate_face_profile
        logger.warning("thumbnail_selection.face_detect_failed", frame=path, error=str(exc))
        has_face = False

    return FrameCandidate(
        path=path, brightness=brightness, sharpness=sharpness, has_face=has_face, is_dead_frame=is_dead_frame
    )


def pick_best_frame(frame_paths: list[str]) -> str | None:
    """Given several candidate frame image paths sampled from one rendered
    clip, return the path of the best one to use as its thumbnail, or None
    if every candidate was unreadable (caller should fall back to the
    original fixed-frame behavior in that case -- this function never
    raises).

    Frames flagged `is_dead_frame` (near-black/near-white -- almost always
    a transition, not real content) are excluded whenever at least one
    non-dead candidate exists; if literally every sampled frame looks dead
    (a genuinely dark clip, or every sample landed on a transition), falls
    back to ranking the full set anyway -- a mediocre real frame beats
    returning nothing.
    """
    analyzed = [c for c in (_analyze_frame(p) for p in frame_paths) if c is not None]
    if not analyzed:
        return None

    viable = [c for c in analyzed if not c.is_dead_frame]
    pool = viable if viable else analyzed

    max_sharpness = max(c.sharpness for c in pool)
    scored = []
    for c in pool:
        normalized_sharpness = (c.sharpness / max_sharpness) if max_sharpness > 0 else 0.0
        composite = _SHARPNESS_WEIGHT * normalized_sharpness + (_FACE_PRESENCE_BONUS if c.has_face else 0.0)
        scored.append(FrameCandidate(c.path, c.brightness, c.sharpness, c.has_face, c.is_dead_frame, composite))

    best = max(scored, key=lambda c: c.composite)
    return best.path
