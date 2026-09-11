"""Real end-to-end test of app.core.stt.local_whisper against real speech
audio and the real 'tiny' Whisper model.

Skips (rather than failing) if the model can't be downloaded -- the first
call to WhisperModel() fetches weights from Hugging Face Hub, which isn't
reachable from every environment (e.g. this repo's CI sandbox has no route
to huggingface.co). On a normal dev machine with internet access this test
runs for real and is the one true confirmation that the local/open-source
STT path actually works end to end -- run it manually after `pip install`
if you want that confirmation locally: `pytest tests/test_local_whisper_provider.py -v`.
"""
import shutil
import subprocess
import sys
import types

import pytest

from app.core.stt.base import PermanentSttError


@pytest.fixture
def speech_wav(tmp_path):
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng not available on PATH (only needed to synthesize test speech)")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    raw_path = tmp_path / "speech_raw.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en", "-w", str(raw_path), "hello world, this is a test"],
        capture_output=True,
        check=True,
    )
    wav_path = tmp_path / "speech.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path), "-ac", "1", "-ar", "16000", str(wav_path)],
        capture_output=True,
        check=True,
    )
    return str(wav_path)


def test_local_whisper_transcribes_real_speech(speech_wav, monkeypatch):
    monkeypatch.setattr("app.core.config.settings.stt_local_model_size", "tiny")
    from app.core.stt.local_whisper import LocalWhisperProvider

    provider = LocalWhisperProvider()
    try:
        result = provider.transcribe(speech_wav)
    except PermanentSttError as exc:
        pytest.skip(f"could not run local Whisper (likely no network to download model weights): {exc}")

    assert result.segments, "expected at least one segment from clear synthetic speech"
    # Tolerant match -- tiny model + robotic TTS voice won't be word-perfect,
    # but should get at least one of these simple, clearly-spoken words.
    lowered = result.full_text.lower()
    assert any(word in lowered for word in ("hello", "world", "test"))


# ---- per-job model size override (StreamJob.stt_model_size) -- no network
# needed, faster_whisper itself is faked out entirely below. ----


def test_provider_model_size_defaults_to_settings(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.stt_local_model_size", "base")
    from app.core.stt.local_whisper import LocalWhisperProvider

    assert LocalWhisperProvider().model_size == "base"
    assert LocalWhisperProvider(model_size=None).model_size == "base"
    assert LocalWhisperProvider(model_size="tiny").model_size == "tiny"


def test_get_model_caches_per_model_size(monkeypatch):
    """_get_model's cache key includes model_size so a worker process that
    handles jobs with different per-job stt_model_size overrides (see
    StreamJob.stt_model_size) keeps every size it's loaded resident instead
    of reloading from disk each time the size changes between jobs."""
    import app.core.stt.local_whisper as local_whisper_module

    calls = []

    class _FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            calls.append(model_size)

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_FakeWhisperModel)
    )
    monkeypatch.setattr(local_whisper_module, "_model_cache", {})

    local_whisper_module._get_model("tiny")
    local_whisper_module._get_model("tiny")  # cached -- must not construct again
    local_whisper_module._get_model("small")  # different size -- new construction

    assert calls == ["tiny", "small"]


def test_get_stt_provider_threads_model_size_for_local_provider(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.stt_provider", "local")
    from app.core.stt import get_stt_provider

    provider = get_stt_provider(model_size="medium")
    assert provider.model_size == "medium"
