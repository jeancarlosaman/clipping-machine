"""Best-effort face detection to bias the vertical crop toward where a
streamer's face/webcam actually is, instead of always cropping dead center
(see app.core.rendering_logic.compute_crop_offset), and to distinguish a
"reacting" layout (small corner webcam + separate main content) from an
"IRL" layout (one handheld/selfie camera filling the frame) -- see
app.core.rendering_logic.classify_reaction_layout and app.workers.rendering
for how that split gets built.

Deliberately NOT motion tracking: one static face profile is estimated per
clip from a few sampled frames of that clip's own window, not re-evaluated
frame-by-frame -- moving the crop mid-clip is real added complexity (the
output would visibly jump/pan) for a signal this MVP doesn't need yet.

Uses OpenCV's bundled Haar cascade (haarcascade_frontalface_default.xml) --
no model download, no GPU, a few ms per frame on CPU, consistent with the
project's "deterministic/cheap signal first" approach elsewhere (see
app.core.scoring_logic). It is not a general person/subject detector: a
streamer facing away from the camera, a stream with no webcam at all, or a
webcam that's just small/dark, will simply find nothing. That is always a
safe outcome -- estimate_face_profile returns None and callers fall back to
the prior centered-crop behavior, never something worse than before this
feature existed.

requirements.txt pins opencv-python to a version that still ships
CascadeClassifier + the bundled Haar cascade XML files (opencv-python 5.x
dropped the legacy cv2.data.haarcascades/CascadeClassifier objdetect API in
favor of DNN-based detectors) -- see that pin's comment before bumping it.
"""
from __future__ import annotations

import statistics

from app.workers.common import logger

_cascade = None  # lazy-loaded singleton, same pattern as app.core.stt.local_whisper's model cache


def _get_cascade():
    global _cascade
    if _cascade is None:
        import cv2  # local import: only pull in cv2 if face-aware cropping is actually used

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(path)
        if classifier.empty():
            raise RuntimeError(f"could not load Haar cascade from {path}")
        _cascade = classifier
    return _cascade


def detect_largest_face(frame_path: str) -> dict | None:
    """For the largest detected face in `frame_path`, in that image's own
    pixel coordinates: `{"center": (x, y), "area": float, "frame_size": (w, h)}`,
    or None if no face was found. A missing or unreadable image is also
    treated as "no face found" rather than raised -- this is a best-effort
    enhancement (see module docstring), not a required step. `frame_size`
    rides along so callers (see estimate_face_profile) don't need a second
    way to learn the source frame's dimensions.
    """
    import cv2

    image = cv2.imread(frame_path)
    if image is None:
        return None

    frame_h, frame_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return {"center": (x + w / 2, y + h / 2), "area": float(w * h), "frame_size": (frame_w, frame_h)}


def estimate_face_profile(frame_paths: list[str]) -> dict | None:
    """Aggregate face profile across several sampled frames of one clip:
    `{"center": (x, y), "area": float, "frame_size": (w, h)}`, or None if
    none of the frames had a detectable face. `center`/`area` are the
    median over frames with a detection -- median rather than mean so one
    spurious detection (e.g. a face-shaped icon flashing on screen for a
    single frame) doesn't pull the estimate as far off target as an outlier
    mean would. `frame_size` is taken from whichever detection happened
    first (all sampled frames come from the same clip/source video, so it
    should be identical across all of them).
    """
    detections: list[dict] = []
    for path in frame_paths:
        try:
            found = detect_largest_face(path)
        except Exception as exc:  # defensive: a corrupt frame/cv2 error is "no signal", not a hard failure
            logger.warning("face_detect.frame_failed", frame=path, error=str(exc))
            found = None
        if found is not None:
            detections.append(found)

    if not detections:
        return None

    xs = [d["center"][0] for d in detections]
    ys = [d["center"][1] for d in detections]
    areas = [d["area"] for d in detections]
    return {
        "center": (statistics.median(xs), statistics.median(ys)),
        "area": statistics.median(areas),
        "frame_size": detections[0]["frame_size"],
    }
