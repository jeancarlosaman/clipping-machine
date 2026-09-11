"""Unit tests for app.core.llm_segmentation -- the OpenAI-calling half of
the LLM segment-suggestion feature. Mocks the `openai` package's client
exactly like test_caption_generation.py does (same fake-client classes),
since both modules share the same call shape on purpose.
"""
import openai as openai_module
import pytest

from app.core.config import settings
from app.core.llm_segmentation import generate_llm_segment_suggestions

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "This is insane, wow!"},
    {"start": 2.5, "end": 4.0, "text": "Can you believe that?"},
    {"start": 4.1, "end": 6.0, "text": "That was hilarious and unreal."},
]


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
def _feature_enabled_with_key(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_segment_suggestions", True)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-fake")
    monkeypatch.setattr(settings, "caption_llm_provider", "openai")


def test_disabled_returns_empty_list(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_segment_suggestions", False)
    assert generate_llm_segment_suggestions(SEGMENTS, 1.0, 90.0) == []


def test_empty_transcript_returns_empty_list():
    assert generate_llm_segment_suggestions([], 1.0, 90.0) == []


def test_no_api_key_returns_empty_list(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    assert generate_llm_segment_suggestions(SEGMENTS, 1.0, 90.0) == []


def test_transcript_too_long_returns_empty_list_without_calling_llm(monkeypatch):
    monkeypatch.setattr(settings, "llm_segment_max_transcript_segments", 2)

    def _fail(**kwargs):
        raise AssertionError("should not have called the LLM client at all")

    monkeypatch.setattr(openai_module, "OpenAI", _fail)
    assert generate_llm_segment_suggestions(SEGMENTS, 1.0, 90.0) == []


def test_happy_path_returns_parsed_suggestions(monkeypatch):
    raw = '[{"start_index": 0, "end_index": 1, "reason": "Big reaction moment."}]'
    monkeypatch.setattr(openai_module, "OpenAI", lambda **kwargs: _FakeClient(content=raw))

    result = generate_llm_segment_suggestions(SEGMENTS, min_len=1.0, max_len=90.0)

    assert result == [{"start": 0.0, "end": 4.0, "reason": "Big reaction moment."}]


def test_malformed_response_degrades_to_empty_list(monkeypatch):
    monkeypatch.setattr(openai_module, "OpenAI", lambda **kwargs: _FakeClient(content="not valid json"))
    assert generate_llm_segment_suggestions(SEGMENTS, 1.0, 90.0) == []


def test_api_exception_degrades_to_empty_list(monkeypatch):
    monkeypatch.setattr(
        openai_module, "OpenAI", lambda **kwargs: _FakeClient(exc=RuntimeError("connection reset"))
    )
    assert generate_llm_segment_suggestions(SEGMENTS, 1.0, 90.0) == []


def test_ollama_provider_uses_local_base_url_and_needs_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "caption_llm_provider", "ollama")
    monkeypatch.setattr(settings, "caption_ollama_model", "llama3.1:8b")
    monkeypatch.setattr(settings, "caption_ollama_base_url", "http://localhost:11434/v1")
    raw = '[{"start_index": 0, "end_index": 0, "reason": "ok"}]'
    seen_kwargs = {}

    def fake_openai_client(**kwargs):
        seen_kwargs.update(kwargs)
        return _FakeClient(content=raw)

    monkeypatch.setattr(openai_module, "OpenAI", fake_openai_client)

    result = generate_llm_segment_suggestions(SEGMENTS, min_len=0.5, max_len=90.0)

    assert result == [{"start": 0.0, "end": 2.0, "reason": "ok"}]
    assert seen_kwargs["base_url"] == "http://localhost:11434/v1"
    assert seen_kwargs["api_key"]


def test_respects_configured_max_suggestions(monkeypatch):
    monkeypatch.setattr(settings, "llm_segment_max_suggestions", 1)
    raw = (
        '[{"start_index": 0, "end_index": 0, "reason": "a"}, '
        '{"start_index": 1, "end_index": 1, "reason": "b"}]'
    )
    monkeypatch.setattr(openai_module, "OpenAI", lambda **kwargs: _FakeClient(content=raw))

    result = generate_llm_segment_suggestions(SEGMENTS, min_len=0.5, max_len=90.0)

    assert len(result) == 1
