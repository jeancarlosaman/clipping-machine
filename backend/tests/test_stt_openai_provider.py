"""Tests app.core.stt.openai_api against a fake OpenAI client -- no network
call, no real API key needed. Covers the two things that are actually this
module's own logic (the SDK call itself is OpenAI's problem, not ours):
chunk-boundary offset merging, and transient-vs-permanent error routing.
"""
import shutil
import subprocess

import httpx
import openai
import pytest

from app.core.stt.base import PermanentSttError, TransientSttError
from app.core.stt.openai_api import OpenAiSttProvider


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeResponse:
    def __init__(self, text, segments, language="en"):
        self.text = text
        self.segments = segments
        self.language = language


class _FakeTranscriptions:
    """Returns one canned response per call, in order -- one per chunk."""

    def __init__(self, responses=None, error=None):
        self._responses = list(responses or [])
        self._error = error
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self._error:
            raise self._error
        return self._responses.pop(0)


class _FakeAudio:
    def __init__(self, transcriptions):
        self.transcriptions = transcriptions


class _FakeClient:
    def __init__(self, transcriptions):
        self.audio = _FakeAudio(transcriptions)


def _provider_with_fake_client(monkeypatch, transcriptions):
    monkeypatch.setattr("app.core.stt.openai_api.settings.openai_api_key", "test-key")
    provider = OpenAiSttProvider()
    provider._client = _FakeClient(transcriptions)
    return provider


@pytest.fixture
def short_wav(tmp_path):
    """~1s of tone -- short enough that plan_chunk_boundaries returns []
    for any target above 1s, and long enough to force chunking when the
    provider's chunk_seconds setting is forced very small (see test below).
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")
    path = tmp_path / "short.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ac", "1", "-ar", "16000", str(path)],
        capture_output=True,
        check=True,
    )
    return str(path)


def test_no_chunking_single_call_returns_segments_unmodified(short_wav, monkeypatch):
    monkeypatch.setattr("app.core.stt.openai_api.settings.stt_openai_chunk_seconds", 600.0)
    responses = [_FakeResponse("hello", [_FakeSegment(0.0, 1.0, "hello")])]
    provider = _provider_with_fake_client(monkeypatch, _FakeTranscriptions(responses))

    result = provider.transcribe(short_wav)

    assert result.full_text == "hello"
    assert [(s.start, s.end, s.text) for s in result.segments] == [(0.0, 1.0, "hello")]


def test_chunked_transcription_offsets_segments_by_chunk_start(short_wav, monkeypatch):
    # Force chunking on a ~1s file by setting an artificially tiny chunk size.
    monkeypatch.setattr("app.core.stt.openai_api.settings.stt_openai_chunk_seconds", 0.4)
    responses = [
        _FakeResponse("first", [_FakeSegment(0.0, 0.3, "first")]),
        _FakeResponse("second", [_FakeSegment(0.0, 0.2, "second")]),
    ]
    transcriptions = _FakeTranscriptions(responses)
    provider = _provider_with_fake_client(monkeypatch, transcriptions)

    result = provider.transcribe(short_wav)

    assert transcriptions.calls == 2
    assert result.full_text == "first second"
    starts = [s.start for s in result.segments]
    # Second chunk's segment start must be offset by its chunk's start time
    # in the original file, not left as the raw 0.0 the fake API returned.
    assert starts[0] == pytest.approx(0.0, abs=0.01)
    assert starts[1] > 0.1


def test_missing_api_key_is_permanent_error(monkeypatch):
    monkeypatch.setattr("app.core.stt.openai_api.settings.openai_api_key", "")
    with pytest.raises(PermanentSttError):
        OpenAiSttProvider()


def test_5xx_status_is_transient(short_wav, monkeypatch):
    monkeypatch.setattr("app.core.stt.openai_api.settings.stt_openai_chunk_seconds", 600.0)
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response = httpx.Response(503, request=request)
    error = openai.APIStatusError("service unavailable", response=response, body=None)
    provider = _provider_with_fake_client(monkeypatch, _FakeTranscriptions(error=error))

    with pytest.raises(TransientSttError):
        provider.transcribe(short_wav)


def test_4xx_status_is_permanent(short_wav, monkeypatch):
    monkeypatch.setattr("app.core.stt.openai_api.settings.stt_openai_chunk_seconds", 600.0)
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response = httpx.Response(400, request=request)
    error = openai.APIStatusError("bad request", response=response, body=None)
    provider = _provider_with_fake_client(monkeypatch, _FakeTranscriptions(error=error))

    with pytest.raises(PermanentSttError):
        provider.transcribe(short_wav)


def test_connection_error_is_transient(short_wav, monkeypatch):
    monkeypatch.setattr("app.core.stt.openai_api.settings.stt_openai_chunk_seconds", 600.0)
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    error = openai.APIConnectionError(request=request)
    provider = _provider_with_fake_client(monkeypatch, _FakeTranscriptions(error=error))

    with pytest.raises(TransientSttError):
        provider.transcribe(short_wav)
