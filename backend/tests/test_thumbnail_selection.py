"""Unit tests for app.core.thumbnail_selection. Sharpness/brightness are
exercised with real cv2 calls against synthetic images (genuine, not
mocked, same spirit as test_face_detect.py) -- face presence is
monkeypatched for determinism, since Haar cascade behavior on synthetic
noise/gradient images isn't something worth depending on.
"""
import numpy as np
import pytest

import app.core.thumbnail_selection as thumbnail_selection
from app.core.thumbnail_selection import pick_best_frame

cv2 = pytest.importorskip("cv2")


def _write_image(path, array) -> str:
    cv2.imwrite(str(path), array)
    return str(path)


def _flat_image(tmp_path, name: str, value: int, size: int = 200) -> str:
    """A single flat color -- zero Laplacian variance (no edges at all)."""
    array = np.full((size, size, 3), value, dtype="uint8")
    return _write_image(tmp_path / name, array)


def _textured_image(tmp_path, name: str, base_value: int, seed: int, size: int = 200) -> str:
    """Real pixel-level texture (a deterministic pseudo-random speckle
    pattern) around a given brightness -- gives a genuinely nonzero,
    reproducible Laplacian variance without needing a real photo."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-40, 40, size=(size, size), endpoint=True)
    channel = np.clip(base_value + noise, 0, 255).astype("uint8")
    array = np.stack([channel, channel, channel], axis=-1)
    return _write_image(tmp_path / name, array)


@pytest.fixture(autouse=True)
def _no_faces_by_default(monkeypatch):
    # Deterministic default for every test in this file -- individual tests
    # override this when they specifically want to test face-presence
    # behavior.
    monkeypatch.setattr(thumbnail_selection, "detect_largest_face", lambda path: None)


def test_pick_best_frame_returns_none_when_every_candidate_is_unreadable():
    assert pick_best_frame(["/tmp/does-not-exist-1.jpg", "/tmp/does-not-exist-2.jpg"]) is None


def test_pick_best_frame_skips_unreadable_candidates(tmp_path):
    real = _textured_image(tmp_path, "real.jpg", base_value=120, seed=1)
    assert pick_best_frame(["/tmp/does-not-exist.jpg", real]) == real


def test_pick_best_frame_prefers_sharper_frame_over_flat_frame(tmp_path):
    flat = _flat_image(tmp_path, "flat.jpg", value=120)
    sharp = _textured_image(tmp_path, "sharp.jpg", base_value=120, seed=2)
    assert pick_best_frame([flat, sharp]) == sharp


def test_pick_best_frame_prefers_frame_with_visible_face(tmp_path, monkeypatch):
    # Two frames with identical texture/sharpness -- only face presence
    # differs, so the face-presence bonus must be what decides it.
    with_face = _textured_image(tmp_path, "with_face.jpg", base_value=120, seed=3)
    without_face = _textured_image(tmp_path, "without_face.jpg", base_value=120, seed=3)

    def fake_detect(path):
        return {"center": (10.0, 10.0), "area": 100.0, "frame_size": (200, 200)} if path == with_face else None

    monkeypatch.setattr(thumbnail_selection, "detect_largest_face", fake_detect)
    assert pick_best_frame([without_face, with_face]) == with_face


def test_pick_best_frame_excludes_dead_dark_frame_even_if_sharper(tmp_path):
    # A near-black frame with scattered bright speckles has a HIGH raw
    # Laplacian variance (lots of edges) but a very low mean brightness --
    # it should still lose to a plainer, viable mid-brightness frame,
    # because "dead" (near-black/near-white) frames are excluded from the
    # pool whenever a non-dead candidate exists, regardless of sharpness.
    rng = np.random.default_rng(4)
    speckled_black = np.zeros((200, 200), dtype="uint8")
    ys = rng.integers(0, 200, size=500)
    xs = rng.integers(0, 200, size=500)
    speckled_black[ys, xs] = 255
    dead_dark = _write_image(tmp_path / "dead_dark.jpg", np.stack([speckled_black] * 3, axis=-1))
    assert cv2.imread(dead_dark).mean() < thumbnail_selection._DARK_BRIGHTNESS_THRESHOLD

    viable = _textured_image(tmp_path, "viable.jpg", base_value=110, seed=5)
    assert pick_best_frame([dead_dark, viable]) == viable


def test_pick_best_frame_excludes_dead_bright_frame_even_if_sharper(tmp_path):
    rng = np.random.default_rng(6)
    speckled_white = np.full((200, 200), 255, dtype="uint8")
    ys = rng.integers(0, 200, size=500)
    xs = rng.integers(0, 200, size=500)
    speckled_white[ys, xs] = 0
    dead_bright = _write_image(tmp_path / "dead_bright.jpg", np.stack([speckled_white] * 3, axis=-1))
    assert cv2.imread(dead_bright).mean() > thumbnail_selection._BRIGHT_BRIGHTNESS_THRESHOLD

    viable = _textured_image(tmp_path, "viable2.jpg", base_value=140, seed=7)
    assert pick_best_frame([dead_bright, viable]) == viable


def test_pick_best_frame_falls_back_to_dead_pool_when_nothing_else_viable(tmp_path):
    # Every sampled frame happens to be dead (e.g. a genuinely dark clip,
    # or every sample landed on a transition) -- still returns a real
    # frame rather than None, since a mediocre real thumbnail beats no
    # thumbnail at all.
    only_dead = _flat_image(tmp_path, "only_dead.jpg", value=0)
    assert pick_best_frame([only_dead]) == only_dead


def test_pick_best_frame_survives_face_detection_raising(tmp_path, monkeypatch):
    frame = _textured_image(tmp_path, "frame.jpg", base_value=120, seed=8)

    def raising_detect(path):
        raise RuntimeError("cv2 exploded")

    monkeypatch.setattr(thumbnail_selection, "detect_largest_face", raising_detect)
    # Should degrade to "no face" for that frame, not propagate the error.
    assert pick_best_frame([frame]) == frame
