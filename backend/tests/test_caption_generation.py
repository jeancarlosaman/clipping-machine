"""Unit tests for app.core.caption_generation -- the OpenAI-calling half of
caption/hashtag generation. The real `openai` package is already a project
dependency (used by app.core.stt.openai_api); these tests monkeypatch its
`OpenAI` client class rather than making real network calls, same spirit as
test_local_whisper_provider.py mocking faster-whisper.
"""
import openai as openai_module
import pytest

from app.core.caption_generation import generate_caption_annotation
from app.core.config import settings

SEGMENTS = [{"start": 0.0, "end": 2.0, "text": "This is insane, wow!"}]
BREAKDOWN = {"contributions": {"emotional_language": 1.2, "speech_density": 0.1}}


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content, exc=None):
        self._content = content
        self._exc = exc

    def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content, exc=None):
        self.completions = _FakeCompletions(content, exc)


class _FakeClient:
    def __init__(self, content=None, exc=None, **kwargs):
        self.chat = _FakeChat(content, exc)


@pytest.fixture(autouse=True)
def _llm_enabled_with_key(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_captions", True)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-fake")


def test_generate_caption_annotation_disabled_returns_heuristic(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_captions", False)
    annotation = generate_caption_annotation(SEGMENTS, 0.0, 2.0, BREAKDOWN, "existing caption")
    assert annotation["source"] == "heuristic_fallback"
    assert annotation["reason"] == "disabled"
    assert annotation["caption"] == "existing caption"
    assert annotation["title"]  # heuristic_title never returns empty
    assert len(annotation["hashtags"]) >= 3


def test_generate_caption_annotation_no_api_key_returns_heuristic(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    annotation = generate_caption_annotation(SEGMENTS, 0.0, 2.0, BREAKDOWN, "existing caption")
    assert annotation["source"] == "heuristic_fallback"
    assert annotation["reason"] == "no_api_key"
    assert annotation["title"]


def test_generate_caption_annotation_happy_path(monkeypatch):
    raw = (
        '{"title": "He Went INSANE", "hashtags": ["insane", "clutch"], '
        '"caption": "He went insane!", "explanation": "Huge reaction moment."}'
    )
    monkeypatch.setattr(openai_module, "OpenAI", lambda **kwargs: _FakeClient(content=raw))

    annotation = generate_caption_annotation(SEGMENTS, 0.0, 2.0, BREAKDOWN, "fallback caption")

    assert annotation["source"] == "llm"
    assert annotation["title"] == "He Went INSANE"
    assert annotation["hashtags"] == ["#insane", "#clutch"]
    assert annotation["caption"] == "He went insane!"
    assert annotation["explanation"] == "Huge reaction moment."
    assert annotation["model"] == settings.caption_llm_model


def test_generate_caption_annotation_falls_back_on_malformed_response(monkeypatch):
    monkeypatch.setattr(openai_module, "OpenAI", lambda **kwargs: _FakeClient(content="not valid json"))

    annotation = generate_caption_annotation(SEGMENTS, 0.0, 2.0, BREAKDOWN, "fallback caption")

    assert annotation["source"] == "heuristic_fallback"
    assert annotation["caption"] == "fallback caption"
    assert "llm_error" in annotation["reason"]


def test_generate_caption_annotation_falls_back_on_api_exception(monkeypatch):
    monkeypatch.setattr(
        openai_module, "OpenAI", lambda **kwargs: _FakeClient(exc=RuntimeError("connection reset"))
    )

    annotation = generate_caption_annotation(SEGMENTS, 0.0, 2.0, BREAKDOWN, "fallback caption")

    assert annotation["source"] == "heuristic_fallback"
    assert "connection reset" in annotation["reason"]


def test_generate_caption_annotation_ollama_happy_path_uses_local_base_url(monkeypatch):
    # No openai_api_key needed for the ollama provider -- confirms the
    # "no_api_key" heuristic gate above only applies to provider="openai".
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "caption_llm_provider", "ollama")
    monkeypatch.setattr(settings, "caption_ollama_model", "llama3.1:8b")
    monkeypatch.setattr(settings, "caption_ollama_base_url", "http://localhost:11434/v1")
    raw = (
        '{"title": "Local Model Title", "hashtags": ["local", "test"], '
        '"caption": "Ran on Ollama.", "explanation": "Grounded in the transcript."}'
    )
    seen_kwargs = {}

    def fake_openai_client(**kwargs):
        seen_kwargs.update(kwargs)
        return _FakeClient(content=raw)

    monkeypatch.setattr(openai_module, "OpenAI", fake_openai_client)

    annotation = generate_caption_annotation(SEGMENTS, 0.0, 2.0, BREAKDOWN, "fallback caption")

    assert annotation["source"] == "llm"
    assert annotation["title"] == "Local Model Title"
    assert annotation["model"] == "llama3.1:8b"
    # Confirms the client was pointed at Ollama's endpoint, not OpenAI's default.
    assert seen_kwargs["base_url"] == "http://localhost:11434/v1"
    assert seen_kwargs["api_key"]  # non-empty placeholder -- the openai client requires *some* string


def test_generate_caption_annotation_ollama_unreachable_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "caption_llm_provider", "ollama")
    monkeypatch.setattr(
        openai_module, "OpenAI", lambda **kwargs: _FakeClient(exc=ConnectionError("connection refused"))
    )

    annotation = generate_caption_annotation(SEGMENTS, 0.0, 2.0, BREAKDOWN, "fallback caption")

    assert annotation["source"] == "heuristic_fallback"
    assert "connection refused" in annotation["reason"]
