"""Thin HTTP client for the official TikTok Content Posting API -- inbox
(draft) upload flow only, per the project's hard MVP constraint (draft
target, no direct publish -- see architecture doc §6). Nothing here decides
*whether* to call TikTok or what to do with the result; that's
app.workers.upload / app.api.routers.creator_accounts. This module only
knows how to make one correct HTTP call at a time and translate TikTok's
response into either a plain dict or a TikTokAPIError.

Endpoints used (all under https://open.tiktokapis.com, except the
browser-facing authorize page which is on www.tiktok.com):
  - POST /v2/oauth/token/                        -- code/refresh_token -> access token
  - POST /v2/post/publish/inbox/video/init/       -- start an inbox (draft) upload, scope video.upload
  - PUT  {upload_url}                             -- send the actual video bytes, one call per chunk
  - POST /v2/post/publish/status/fetch/           -- poll a publish_id's status

Deliberately NOT implemented: /v2/post/publish/video/init/ (Direct Post,
publishes straight to the feed) -- that's the post-MVP "direct" target_mode,
already refused at the API layer (app.schemas.UploadRequest) before it
would ever reach here.
"""
from __future__ import annotations

from urllib.parse import urlencode

import requests

from app.core.config import settings

AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
INIT_INBOX_UPLOAD_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
STATUS_FETCH_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"

# video.upload (not video.publish) -- the inbox/draft scope, matching the
# init endpoint above. video.publish is Direct Post's scope; requesting it
# would be asking a creator to grant more than this app actually uses.
OAUTH_SCOPE = "video.upload"

# TikTok's own per-chunk bounds for the FILE_UPLOAD source (except when the
# whole video is the only chunk, which has no floor -- see plan_chunks).
MIN_CHUNK_SIZE_BYTES = 5_000_000
MAX_CHUNK_SIZE_BYTES = 64_000_000

TERMINAL_SUCCESS_STATUSES = {"SEND_TO_USER_INBOX", "PUBLISH_COMPLETE"}
TERMINAL_FAILURE_STATUS = "FAILED"

_JSON_TIMEOUT_SECONDS = 30
# Chunk PUTs move real video bytes, not a small JSON payload -- generous
# fixed timeout rather than trying to scale it per chunk size for an MVP
# where chunks are usually small vertical clips anyway.
_UPLOAD_TIMEOUT_SECONDS = 300


class TikTokAPIError(RuntimeError):
    """Raised for any TikTok API call that didn't succeed.

    `retryable` is the one field callers actually branch on (see
    app.workers.upload's module docstring for the retry contract this
    exists to support): True for network errors, timeouts, and 5xx --
    False for 4xx/policy rejections and TikTok's own non-"ok" error
    envelope, none of which a retry would fix.
    """

    def __init__(self, message: str, *, status_code: int | None = None, error_code: str | None = None, retryable: bool):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable


def build_authorize_url(*, redirect_uri: str, state: str) -> str:
    """The URL to send a creator's browser to for TikTok's consent screen.
    Not fetched from here -- the caller (creator_accounts.tiktok_oauth_start)
    hands this back to the dev console, which does the actual redirect."""
    params = {
        "client_key": settings.tiktok_client_key,
        "response_type": "code",
        "scope": OAUTH_SCOPE,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _post_json(url: str, *, access_token: str | None, json_body: dict) -> dict:
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    try:
        resp = requests.post(url, json=json_body, headers=headers, timeout=_JSON_TIMEOUT_SECONDS)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise TikTokAPIError(f"network error calling {url}: {exc}", retryable=True) from exc

    return _unwrap_envelope(resp, url)


def _unwrap_envelope(resp: requests.Response, url: str) -> dict:
    # TikTok returns a 200 with an `error.code` field even for most
    # application-level failures (bad token, bad request shape, ...) --
    # HTTP status alone isn't enough to tell success from failure here, but
    # it *is* how we tell "TikTok is having a bad day" (5xx -- retry) apart
    # from "this request is wrong" (4xx -- don't retry the same thing).
    if resp.status_code >= 500:
        raise TikTokAPIError(
            f"TikTok API {url} returned {resp.status_code}", status_code=resp.status_code, retryable=True
        )

    try:
        body = resp.json()
    except ValueError as exc:
        raise TikTokAPIError(
            f"TikTok API {url} returned non-JSON body (status {resp.status_code})",
            status_code=resp.status_code,
            retryable=resp.status_code >= 500,
        ) from exc

    error = body.get("error") or {}
    error_code = error.get("code")
    if resp.status_code >= 400 or (error_code and error_code != "ok"):
        raise TikTokAPIError(
            f"TikTok API {url} error: {error_code or resp.status_code} -- {error.get('message', '')}",
            status_code=resp.status_code,
            error_code=error_code,
            retryable=False,
        )

    return body.get("data", {})


def exchange_code_for_token(code: str, redirect_uri: str) -> dict:
    """Authorization-code grant. Returns TikTok's token response dict
    (access_token, open_id, refresh_token, expires_in, refresh_expires_in,
    scope, token_type) -- the caller persists whatever of that it needs."""
    return _oauth_token_request(
        {
            "client_key": settings.tiktok_client_key,
            "client_secret": settings.tiktok_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
    )


def refresh_access_token(refresh_token: str) -> dict:
    """Refresh-token grant. Same response shape as exchange_code_for_token."""
    return _oauth_token_request(
        {
            "client_key": settings.tiktok_client_key,
            "client_secret": settings.tiktok_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    )


def _oauth_token_request(form_body: dict) -> dict:
    # The token endpoint is form-encoded, not JSON, per TikTok's docs --
    # deliberately not routed through _post_json, which always sends JSON.
    try:
        resp = requests.post(
            TOKEN_URL,
            data=form_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=_JSON_TIMEOUT_SECONDS,
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise TikTokAPIError(f"network error calling {TOKEN_URL}: {exc}", retryable=True) from exc

    if resp.status_code >= 500:
        raise TikTokAPIError(f"TikTok token endpoint returned {resp.status_code}", status_code=resp.status_code, retryable=True)

    try:
        body = resp.json()
    except ValueError as exc:
        raise TikTokAPIError(
            f"TikTok token endpoint returned non-JSON body (status {resp.status_code})",
            status_code=resp.status_code,
            retryable=False,
        ) from exc

    # The token endpoint's error shape is flatter than the rest of the
    # API's ({error, error_description} at the top level, no {data:...}
    # wrapper) -- handled separately from _unwrap_envelope for that reason.
    if resp.status_code >= 400 or "error" in body:
        raise TikTokAPIError(
            f"TikTok token endpoint error: {body.get('error')} -- {body.get('error_description', '')}",
            status_code=resp.status_code,
            error_code=body.get("error"),
            retryable=False,
        )
    return body


def plan_chunks(video_size_bytes: int, *, preferred_chunk_size: int) -> tuple[int, int]:
    """Return (chunk_size, total_chunk_count) for init_inbox_upload.

    Single chunk (the whole file) when it's under preferred_chunk_size --
    TikTok's 5MB chunk-size floor is explicitly waived "when it is the only
    chunk" per their docs, so a short vertical clip (the common case here)
    doesn't need artificial splitting. Above that, split into
    preferred_chunk_size pieces (clamped into TikTok's allowed 5-64MB
    range), last chunk taking the remainder.
    """
    if video_size_bytes <= preferred_chunk_size:
        return video_size_bytes, 1
    chunk_size = max(MIN_CHUNK_SIZE_BYTES, min(preferred_chunk_size, MAX_CHUNK_SIZE_BYTES))
    total_chunks = (video_size_bytes + chunk_size - 1) // chunk_size
    return chunk_size, total_chunks


def init_inbox_upload(access_token: str, *, video_size: int, chunk_size: int, total_chunk_count: int) -> dict:
    """Starts an inbox (draft) upload. Returns {"publish_id", "upload_url"}."""
    return _post_json(
        INIT_INBOX_UPLOAD_URL,
        access_token=access_token,
        json_body={
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunk_count,
            }
        },
    )


def upload_video_chunk(
    upload_url: str, chunk_bytes: bytes, *, first_byte: int, last_byte: int, total_bytes: int, content_type: str = "video/mp4"
) -> None:
    """PUTs one chunk of the video to the upload_url init_inbox_upload
    returned. No return value -- TikTok signals per-chunk success via HTTP
    status only, nothing to parse out of the body."""
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(chunk_bytes)),
        "Content-Range": f"bytes {first_byte}-{last_byte}/{total_bytes}",
    }
    try:
        resp = requests.put(upload_url, data=chunk_bytes, headers=headers, timeout=_UPLOAD_TIMEOUT_SECONDS)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise TikTokAPIError(f"network error uploading chunk to {upload_url}: {exc}", retryable=True) from exc

    if resp.status_code >= 500:
        raise TikTokAPIError(f"chunk upload returned {resp.status_code}", status_code=resp.status_code, retryable=True)
    if resp.status_code >= 400:
        raise TikTokAPIError(
            f"chunk upload rejected: {resp.status_code} -- {resp.text[:500]}",
            status_code=resp.status_code,
            retryable=False,
        )


def fetch_publish_status(access_token: str, publish_id: str) -> dict:
    """Returns {"status", "fail_reason", "publicaly_available_post_id", ...}
    (field names as TikTok's API actually spells them, typo included)."""
    return _post_json(STATUS_FETCH_URL, access_token=access_token, json_body={"publish_id": publish_id})
