"""Unit tests for app.core.tiktok_client -- the real `requests` package is
already a project dependency; these tests monkeypatch requests.post/put
rather than making real network calls, same spirit as
test_caption_generation.py mocking the OpenAI client.
"""
import pytest
import requests

from app.core import tiktok_client
from app.core.config import settings
from app.core.tiktok_client import TikTokAPIError


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text="", raise_json_error=False):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self._raise_json_error = raise_json_error

    def json(self):
        if self._raise_json_error:
            raise ValueError("not json")
        return self._json_body


@pytest.fixture(autouse=True)
def _tiktok_credentials(monkeypatch):
    monkeypatch.setattr(settings, "tiktok_client_key", "test-client-key")
    monkeypatch.setattr(settings, "tiktok_client_secret", "test-client-secret")


def test_build_authorize_url_includes_required_params():
    url = tiktok_client.build_authorize_url(redirect_uri="https://example.com/cb", state="abc123")
    assert url.startswith(tiktok_client.AUTHORIZE_URL)
    assert "client_key=test-client-key" in url
    assert "response_type=code" in url
    assert "scope=video.upload" in url
    assert "state=abc123" in url
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcb" in url


def test_exchange_code_for_token_success(monkeypatch):
    def fake_post(url, data=None, headers=None, timeout=None):
        assert url == tiktok_client.TOKEN_URL
        assert data["code"] == "auth-code"
        assert data["grant_type"] == "authorization_code"
        return _FakeResponse(200, {"access_token": "at", "open_id": "oid", "expires_in": 86400})

    monkeypatch.setattr(requests, "post", fake_post)
    result = tiktok_client.exchange_code_for_token("auth-code", "https://example.com/cb")
    assert result["access_token"] == "at"
    assert result["open_id"] == "oid"


def test_exchange_code_for_token_rejects_error_response(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _FakeResponse(400, {"error": "invalid_grant", "error_description": "bad code"}),
    )
    with pytest.raises(TikTokAPIError) as exc_info:
        tiktok_client.exchange_code_for_token("bad-code", "https://example.com/cb")
    assert exc_info.value.retryable is False


def test_refresh_access_token_success(monkeypatch):
    def fake_post(url, data=None, headers=None, timeout=None):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "rt"
        return _FakeResponse(200, {"access_token": "new-at", "refresh_token": "new-rt", "expires_in": 86400})

    monkeypatch.setattr(requests, "post", fake_post)
    result = tiktok_client.refresh_access_token("rt")
    assert result["access_token"] == "new-at"


def test_init_inbox_upload_success(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        assert url == tiktok_client.INIT_INBOX_UPLOAD_URL
        assert headers["Authorization"] == "Bearer at-123"
        assert json["source_info"]["source"] == "FILE_UPLOAD"
        return _FakeResponse(200, {"data": {"publish_id": "pub_1", "upload_url": "https://upload.example/x"}, "error": {"code": "ok"}})

    monkeypatch.setattr(requests, "post", fake_post)
    result = tiktok_client.init_inbox_upload("at-123", video_size=1000, chunk_size=1000, total_chunk_count=1)
    assert result == {"publish_id": "pub_1", "upload_url": "https://upload.example/x"}


def test_init_inbox_upload_raises_non_retryable_on_4xx(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _FakeResponse(
            401, {"data": {}, "error": {"code": "access_token_invalid", "message": "bad token"}}
        ),
    )
    with pytest.raises(TikTokAPIError) as exc_info:
        tiktok_client.init_inbox_upload("bad-token", video_size=1000, chunk_size=1000, total_chunk_count=1)
    assert exc_info.value.retryable is False
    assert exc_info.value.error_code == "access_token_invalid"


def test_init_inbox_upload_raises_retryable_on_5xx(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(503, {"error": {}}))
    with pytest.raises(TikTokAPIError) as exc_info:
        tiktok_client.init_inbox_upload("at", video_size=1000, chunk_size=1000, total_chunk_count=1)
    assert exc_info.value.retryable is True


def test_init_inbox_upload_raises_retryable_on_network_error(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "post", raise_conn_error)
    with pytest.raises(TikTokAPIError) as exc_info:
        tiktok_client.init_inbox_upload("at", video_size=1000, chunk_size=1000, total_chunk_count=1)
    assert exc_info.value.retryable is True


def test_upload_video_chunk_sends_correct_headers(monkeypatch):
    captured = {}

    def fake_put(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "put", fake_put)
    tiktok_client.upload_video_chunk(
        "https://upload.example/x", b"chunkbytes", first_byte=0, last_byte=9, total_bytes=20
    )
    assert captured["url"] == "https://upload.example/x"
    assert captured["headers"]["Content-Range"] == "bytes 0-9/20"
    assert captured["headers"]["Content-Length"] == "10"
    assert captured["data"] == b"chunkbytes"


def test_upload_video_chunk_raises_non_retryable_on_4xx(monkeypatch):
    monkeypatch.setattr(requests, "put", lambda *a, **k: _FakeResponse(400, text="bad chunk"))
    with pytest.raises(TikTokAPIError) as exc_info:
        tiktok_client.upload_video_chunk("https://upload.example/x", b"x", first_byte=0, last_byte=0, total_bytes=1)
    assert exc_info.value.retryable is False


def test_fetch_publish_status_success(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _FakeResponse(200, {"data": {"status": "PUBLISH_COMPLETE"}, "error": {"code": "ok"}}),
    )
    result = tiktok_client.fetch_publish_status("at", "pub_1")
    assert result["status"] == "PUBLISH_COMPLETE"


@pytest.mark.parametrize(
    "video_size,preferred,expected_chunk,expected_count",
    [
        (1_000_000, 10_000_000, 1_000_000, 1),  # small file -> single chunk, no 5MB floor applied
        (25_000_000, 10_000_000, 10_000_000, 3),  # 25MB / 10MB chunks -> 3 chunks (last one partial)
    ],
)
def test_plan_chunks(video_size, preferred, expected_chunk, expected_count):
    chunk_size, total_chunks = tiktok_client.plan_chunks(video_size, preferred_chunk_size=preferred)
    assert chunk_size == expected_chunk
    assert total_chunks == expected_count


def test_plan_chunks_clamps_preferred_size_into_tiktok_bounds():
    # preferred_chunk_size below TikTok's 5MB floor gets clamped up, for a
    # file large enough that it isn't just sent as a single chunk.
    chunk_size, _ = tiktok_client.plan_chunks(50_000_000, preferred_chunk_size=1_000_000)
    assert chunk_size == tiktok_client.MIN_CHUNK_SIZE_BYTES
