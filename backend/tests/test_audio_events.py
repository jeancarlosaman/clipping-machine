"""Tests for app.core.audio_events. Split the same way as
test_tiktok_client.py / test_caption_generation.py: real, non-mocked
behavior for anything that's genuinely cheap to run for real (ffmpeg is
already a hard project dependency, so extract_window_waveform is exercised
against a real synthetic audio file, not mocked) -- but panns_inference and
torch are NOT installed in this test environment (they're an opt-in, lazily
-imported dependency, see module docstring), so AudioEventDetector's tests
inject a fake `panns_inference` module via sys.modules rather than
requiring the real ~300MB checkpoint and a torch install just to run the
test suite.
"""
import subprocess
import sys
import types

import numpy as np
import pytest

from app.core import audio_events


# ---- extract_window_waveform: real ffmpeg, no mocking ----


@pytest.fixture(scope="module")
def sine_wav_path(tmp_path_factory):
    """A real 4-second 440Hz tone, generated via ffmpeg's synthetic lavfi
    source -- no fixture audio file needed, and no network call."""
    path = tmp_path_factory.mktemp("audio_events") / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-ac", "1", "-ar", "44100", str(path),
        ],
        check=True,
    )
    return str(path)


def test_extract_window_waveform_real_ffmpeg_shape_and_dtype(sine_wav_path):
    waveform = audio_events.extract_window_waveform(sine_wav_path, 1.0, 2.5)
    assert waveform.dtype == np.float32
    # 1.5s at PANNs' fixed 32kHz -- allow ffmpeg's usual +/- one-frame slop.
    expected_samples = int(1.5 * audio_events.SAMPLE_RATE)
    assert abs(waveform.size - expected_samples) < 200
    assert waveform.max() <= 1.0 and waveform.min() >= -1.0
    assert np.abs(waveform).mean() > 0.01  # a real tone, not silence/zeros


def test_extract_window_waveform_handles_tiny_window(sine_wav_path):
    # start == end would be a zero-duration ffmpeg request -- the `max(...,
    # 0.05)` floor in extract_window_waveform's implementation exists
    # specifically so a degenerate window still returns *something* rather
    # than an empty/failed ffmpeg call.
    waveform = audio_events.extract_window_waveform(sine_wav_path, 1.0, 1.0)
    assert waveform.size > 0


# ---- _download: mocked requests, no real network ----


def test_download_skips_when_file_already_large_enough(tmp_path, monkeypatch):
    path = tmp_path / "existing.bin"
    path.write_bytes(b"x" * 2000)

    def fail_if_called(*a, **k):
        raise AssertionError("should not have made a network call")

    monkeypatch.setattr(audio_events.requests, "get", fail_if_called)
    audio_events._download(path, "http://example.invalid/file", min_bytes=1000)  # must not raise


def test_download_fetches_when_missing(tmp_path, monkeypatch):
    path = tmp_path / "missing.bin"
    calls = []

    class _FakeResponse:
        content = b"y" * 5000

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(audio_events.requests, "get", fake_get)
    audio_events._download(path, "http://example.invalid/file", min_bytes=1000)

    assert calls == ["http://example.invalid/file"]
    assert path.read_bytes() == b"y" * 5000


def test_download_refetches_undersized_file(tmp_path, monkeypatch):
    # Mirrors panns_inference's own "< 3e8 bytes means a prior download was
    # truncated/corrupt, re-fetch" guard (see module docstring) -- our
    # pre-fetch has to agree with that or the two could disagree about
    # whether a file is "really there".
    path = tmp_path / "truncated.bin"
    path.write_bytes(b"x" * 10)  # too small
    calls = []

    class _FakeResponse:
        content = b"z" * 5000

        def raise_for_status(self):
            pass

    monkeypatch.setattr(audio_events.requests, "get", lambda url, timeout: calls.append(url) or _FakeResponse())
    audio_events._download(path, "http://example.invalid/file", min_bytes=1000)

    assert calls == ["http://example.invalid/file"]
    assert path.read_bytes() == b"z" * 5000


# ---- AudioEventDetector: fake panns_inference module, no real model ----


class _FakeAudioTagging:
    """Stand-in for panns_inference.AudioTagging -- records how it was
    constructed and returns a canned clipwise_output shaped like the real
    thing (batch=1, classes=len(labels))."""

    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeAudioTagging.last_kwargs = kwargs

    def inference(self, audio):
        classes_num = len(_fake_panns_module.labels)
        output = np.zeros((audio.shape[0], classes_num), dtype=np.float32)
        # Match the module-level test's expectations: 90% at index 2, 60% at index 5.
        output[0, 2] = 0.9
        output[0, 5] = 0.6
        return output, np.zeros((audio.shape[0], 10), dtype=np.float32)


_fake_panns_module = types.ModuleType("panns_inference")
_fake_panns_module.labels = ["Silence", "Speech", "Laughter", "Music", "Noise", "Cheering", "Other"]
_fake_panns_module.AudioTagging = _FakeAudioTagging


@pytest.fixture
def fake_panns(monkeypatch):
    monkeypatch.setitem(sys.modules, "panns_inference", _fake_panns_module)
    monkeypatch.setattr(audio_events, "_ensure_panns_data", lambda: None)
    monkeypatch.setattr(audio_events, "LAUGHTER_LABELS", ("Laughter",))
    monkeypatch.setattr(audio_events, "CROWD_REACTION_LABELS", ("Cheering",))
    yield


def test_detector_looks_up_labels_by_name_not_hardcoded_index(fake_panns):
    detector = audio_events.AudioEventDetector()
    # "Laughter" is index 2 and "Cheering" is index 5 in the fake label
    # list above -- confirms the lookup is genuinely by name, not assuming
    # any particular fixed index.
    assert detector._laughter_indices == [2]
    assert detector._crowd_indices == [5]


def test_detector_score_window_reads_correct_probabilities(fake_panns, monkeypatch):
    detector = audio_events.AudioEventDetector()
    monkeypatch.setattr(
        audio_events, "extract_window_waveform", lambda path, start, end: np.zeros(1000, dtype=np.float32)
    )
    result = detector.score_window("fake.wav", 0.0, 1.0)
    assert result == {"laughter": pytest.approx(0.9), "crowd_reaction": pytest.approx(0.6)}


def test_detector_missing_label_logs_and_is_not_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "panns_inference", _fake_panns_module)
    monkeypatch.setattr(audio_events, "_ensure_panns_data", lambda: None)
    monkeypatch.setattr(audio_events, "LAUGHTER_LABELS", ("Laughter", "Some Label Not In Fake List"))
    monkeypatch.setattr(audio_events, "CROWD_REACTION_LABELS", ("Cheering",))

    detector = audio_events.AudioEventDetector()  # must not raise despite the missing label
    assert detector._laughter_indices == [2]  # only the real one made it in


def test_detector_score_window_never_raises_on_extraction_failure(fake_panns, monkeypatch):
    detector = audio_events.AudioEventDetector()

    def broken_extract(path, start, end):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(audio_events, "extract_window_waveform", broken_extract)
    result = detector.score_window("fake.wav", 0.0, 1.0)
    assert result == {"laughter": 0.0, "crowd_reaction": 0.0}


# ---- get_audio_event_detector: lazy singleton ----


def test_get_audio_event_detector_constructs_only_once(monkeypatch):
    monkeypatch.setattr(audio_events, "_detector", None)
    calls = []

    class _CountingDetector:
        def __init__(self):
            calls.append(1)

    monkeypatch.setattr(audio_events, "AudioEventDetector", _CountingDetector)

    first = audio_events.get_audio_event_detector()
    second = audio_events.get_audio_event_detector()

    assert first is second
    assert len(calls) == 1
