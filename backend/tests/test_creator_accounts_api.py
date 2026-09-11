import pytest

from app.core import tiktok_client
from app.core.auth import create_oauth_state_token
from app.core.config import settings
from app.core.crypto import decrypt_token
from app.db.models import CreatorAccount


def test_create_creator_account_success(client, auth_headers, db_session, user):
    resp = client.post(
        "/api/v1/creator-accounts",
        headers=auth_headers,
        json={
            "platform": "tiktok",
            "external_account_id": "@streamer",
            "access_token": "real-access-token",
            "refresh_token": "real-refresh-token",
            "daily_upload_cap": 5,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["platform"] == "tiktok"
    assert body["external_account_id"] == "@streamer"
    assert body["daily_upload_cap"] == 5
    # Token values must never be echoed back by the API.
    assert "access_token" not in body
    assert "access_token_encrypted" not in body

    row = db_session.query(CreatorAccount).filter_by(id=body["id"]).one()
    assert row.access_token_encrypted != "real-access-token"
    assert decrypt_token(row.access_token_encrypted) == "real-access-token"
    assert decrypt_token(row.refresh_token_encrypted) == "real-refresh-token"


def test_create_creator_account_without_refresh_token(client, auth_headers):
    resp = client.post(
        "/api/v1/creator-accounts",
        headers=auth_headers,
        json={"platform": "youtube", "external_account_id": "channel-1", "access_token": "tok"},
    )
    assert resp.status_code == 201
    assert resp.json()["daily_upload_cap"] == 3  # settings.default_daily_upload_cap


def test_create_creator_account_rejects_duplicate(client, auth_headers):
    payload = {"platform": "tiktok", "external_account_id": "@dupe", "access_token": "tok"}
    first = client.post("/api/v1/creator-accounts", headers=auth_headers, json=payload)
    assert first.status_code == 201

    second = client.post("/api/v1/creator-accounts", headers=auth_headers, json=payload)
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "creator_account_already_exists"


def test_create_creator_account_rejects_invalid_platform(client, auth_headers):
    resp = client.post(
        "/api/v1/creator-accounts",
        headers=auth_headers,
        json={"platform": "myspace", "external_account_id": "x", "access_token": "tok"},
    )
    assert resp.status_code == 422  # pydantic Literal rejection


def test_create_creator_account_rejects_daily_cap_above_hard_ceiling(client, auth_headers):
    resp = client.post(
        "/api/v1/creator-accounts",
        headers=auth_headers,
        json={"platform": "tiktok", "external_account_id": "@x", "access_token": "tok", "daily_upload_cap": 999},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_daily_upload_cap"


def test_create_creator_account_rejects_daily_cap_below_one(client, auth_headers):
    resp = client.post(
        "/api/v1/creator-accounts",
        headers=auth_headers,
        json={"platform": "tiktok", "external_account_id": "@x", "access_token": "tok", "daily_upload_cap": 0},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_daily_upload_cap"


def test_list_creator_accounts_only_returns_own(client, auth_headers, db_session):
    from app.core.auth import create_access_token
    from app.db.models import User

    other = User(email="other@example.com", password_hash="x")
    db_session.add(other)
    db_session.commit()
    db_session.refresh(other)
    other_headers = {"Authorization": f"Bearer {create_access_token(str(other.id))}"}

    client.post(
        "/api/v1/creator-accounts",
        headers=auth_headers,
        json={"platform": "tiktok", "external_account_id": "@mine", "access_token": "tok"},
    )
    client.post(
        "/api/v1/creator-accounts",
        headers=other_headers,
        json={"platform": "tiktok", "external_account_id": "@theirs", "access_token": "tok"},
    )

    resp = client.get("/api/v1/creator-accounts", headers=auth_headers)
    assert resp.status_code == 200
    ids = [a["external_account_id"] for a in resp.json()]
    assert ids == ["@mine"]


@pytest.fixture(autouse=True)
def _tiktok_app_configured(monkeypatch):
    monkeypatch.setattr(settings, "tiktok_client_key", "test-client-key")
    monkeypatch.setattr(settings, "tiktok_client_secret", "test-client-secret")
    monkeypatch.setattr(settings, "tiktok_redirect_uri", "https://example.com/cb")


def test_tiktok_oauth_start_returns_authorize_url(client, auth_headers, user):
    resp = client.get("/api/v1/creator-accounts/tiktok/oauth/start", headers=auth_headers)
    assert resp.status_code == 200
    url = resp.json()["authorize_url"]
    assert url.startswith(tiktok_client.AUTHORIZE_URL)
    assert "client_key=test-client-key" in url
    assert "state=" in url


def test_tiktok_oauth_start_503s_when_not_configured(client, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "tiktok_client_key", "")
    resp = client.get("/api/v1/creator-accounts/tiktok/oauth/start", headers=auth_headers)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "tiktok_not_configured"


def test_tiktok_oauth_start_requires_auth(client):
    resp = client.get("/api/v1/creator-accounts/tiktok/oauth/start")
    assert resp.status_code == 401


def test_tiktok_oauth_callback_creates_account(client, user, db_session, monkeypatch):
    monkeypatch.setattr(
        tiktok_client, "exchange_code_for_token",
        lambda code, redirect_uri: {
            "access_token": "at-1", "open_id": "open-id-1", "refresh_token": "rt-1",
            "expires_in": 86400, "scope": "video.upload",
        },
    )
    state = create_oauth_state_token(str(user.id))

    resp = client.get("/api/v1/creator-accounts/tiktok/oauth/callback", params={"code": "auth-code", "state": state})

    assert resp.status_code == 200
    assert "Connected" in resp.text

    row = db_session.query(CreatorAccount).filter_by(user_id=user.id, platform="tiktok").one()
    assert row.external_account_id == "open-id-1"
    assert decrypt_token(row.access_token_encrypted) == "at-1"
    assert decrypt_token(row.refresh_token_encrypted) == "rt-1"
    assert row.scopes == ["video.upload"]
    assert row.token_expires_at is not None


def test_tiktok_oauth_callback_upserts_existing_account(client, user, db_session, monkeypatch):
    monkeypatch.setattr(
        tiktok_client, "exchange_code_for_token",
        lambda code, redirect_uri: {"access_token": "at-first", "open_id": "open-id-1", "expires_in": 3600},
    )
    state = create_oauth_state_token(str(user.id))
    first = client.get("/api/v1/creator-accounts/tiktok/oauth/callback", params={"code": "c1", "state": state})
    assert first.status_code == 200

    monkeypatch.setattr(
        tiktok_client, "exchange_code_for_token",
        lambda code, redirect_uri: {"access_token": "at-second", "open_id": "open-id-1", "expires_in": 3600},
    )
    second = client.get("/api/v1/creator-accounts/tiktok/oauth/callback", params={"code": "c2", "state": state})
    assert second.status_code == 200

    rows = db_session.query(CreatorAccount).filter_by(user_id=user.id, platform="tiktok").all()
    assert len(rows) == 1  # updated in place, not a duplicate row
    assert decrypt_token(rows[0].access_token_encrypted) == "at-second"


def test_tiktok_oauth_callback_handles_user_decline(client):
    resp = client.get(
        "/api/v1/creator-accounts/tiktok/oauth/callback",
        params={"error": "access_denied", "error_description": "user cancelled"},
    )
    assert resp.status_code == 400
    assert "declined" in resp.text.lower()


def test_tiktok_oauth_callback_rejects_invalid_state(client):
    resp = client.get(
        "/api/v1/creator-accounts/tiktok/oauth/callback", params={"code": "auth-code", "state": "not-a-real-token"}
    )
    assert resp.status_code == 400
    assert "expired" in resp.text.lower() or "invalid" in resp.text.lower()


def test_tiktok_oauth_callback_surfaces_token_exchange_failure(client, user, monkeypatch):
    def raise_error(code, redirect_uri):
        raise tiktok_client.TikTokAPIError("bad code", retryable=False)

    monkeypatch.setattr(tiktok_client, "exchange_code_for_token", raise_error)
    state = create_oauth_state_token(str(user.id))

    resp = client.get("/api/v1/creator-accounts/tiktok/oauth/callback", params={"code": "bad", "state": state})
    assert resp.status_code == 400
    assert "rejected" in resp.text.lower()
