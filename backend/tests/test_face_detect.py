"""Unit tests for app.core.face_detect. Real face-detection accuracy isn't
tested here (that would need real face photos and would be flaky/slow) --
instead these exercise the aggregation/fallback logic with a real cv2 call
against a nonexistent/blank image (guaranteed "no face found", genuinely
exercised, not mocked) and a monkeypatched detect_largest_face for the
median-aggregation behavior, same spirit as test_local_whisper_provider.py.
"""
import app.core.face_detect as face_detect


def test_detect_largest_face_returns_none_for_missing_file():
    assert face_detect.detect_largest_face("/tmp/does-not-exist-at-all.jpg") is None


def test_detect_largest_face_returns_none_for_blank_image(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "blank.jpg"
    blank = np.zeros((200, 200, 3), dtype="uint8")
    cv2.imwrite(str(path), blank)
    assert face_detect.detect_largest_face(str(path)) is None


def test_estimate_face_profile_returns_none_when_no_faces_found(monkeypatch):
    monkeypatch.setattr(face_detect, "detect_largest_face", lambda path: None)
    assert face_detect.estimate_face_profile(["a.jpg", "b.jpg", "c.jpg"]) is None


def test_estimate_face_profile_returns_none_for_empty_input():
    assert face_detect.estimate_face_profile([]) is None


def test_estimate_face_profile_medians_across_frames_with_detections(monkeypatch):
    results = {
        "a.jpg": {"center": (100.0, 200.0), "area": 500.0, "frame_size": (1920, 1080)},
        "b.jpg": None,  # no face in this frame -- should be ignored, not treated as (0,0)
        "c.jpg": {"center": (140.0, 220.0), "area": 700.0, "frame_size": (1920, 1080)},
    }
    monkeypatch.setattr(face_detect, "detect_largest_face", lambda path: results[path])
    profile = face_detect.estimate_face_profile(["a.jpg", "b.jpg", "c.jpg"])
    assert profile == {"center": (120.0, 210.0), "area": 600.0, "frame_size": (1920, 1080)}


def test_estimate_face_profile_survives_one_frame_raising(monkeypatch):
    # A corrupt frame/cv2 error on one sample must not take down the whole
    # estimate -- it's just treated as "no face in that frame".
    def flaky(path):
        if path == "bad.jpg":
            raise RuntimeError("corrupt frame")
        return {"center": (100.0, 100.0), "area": 500.0, "frame_size": (640, 360)}

    monkeypatch.setattr(face_detect, "detect_largest_face", flaky)
    profile = face_detect.estimate_face_profile(["bad.jpg", "good.jpg"])
    assert profile == {"center": (100.0, 100.0), "area": 500.0, "frame_size": (640, 360)}
