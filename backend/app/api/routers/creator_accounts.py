"""GET /api/v1/creator-accounts, TikTok OAuth start + callback -- architecture
doc §6.

Real TikTok OAuth, not a stub: `tiktok_oauth_start` (authenticated -- needs
CurrentUserDep to know who's connecting) mints the authorize URL + a signed
`state`; `tiktok_oauth_callback` is where TikTok's redirect actually lands
(a plain browser GET, unauthenticated -- no bearer header exists for a
server-to-server-looking-but-actually-browser redirect), decodes `state`
back into a user_id (app.core.auth.create_oauth_state_token /
decode_oauth_state_token), exchanges `code` for tokens via
app.core.tiktok_client, and upserts a CreatorAccount row -- same shape a
manually-pasted-token account already has, so app.workers.upload doesn't
need to know which path created a given row.

Requires a registered TikTok developer app (Content Posting API product,
Direct Post enabled) with TIKTOK_CLIENT_KEY/TIKTOK_CLIENT_SECRET/
TIKTOK_REDIRECT_UI set -- see architecture doc §10 risks; app review/
domain verification is a separate, TikTok-side lead-time item, not
something this code can shortcut.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import CurrentUserDep, DbDep
from app.api.errors import ApiError
from app.core import tiktok_client
from app.core.auth import create_oauth_state_token, decode_oauth_state_token
from app.core.config import settings
from app.core.crypto import encrypt_token
from app.core.logging import get_logger
from app.db.models import CreatorAccount, User
from app.schemas import CreatorAccountCreate, CreatorAccountOut, TikTokOAuthStartOut

router = APIRouter(prefix="/api/v1/creator-accounts", tags=["creator-accounts"])
log = get_logger(__name__)


@router.get("", response_model=list[CreatorAccountOut])
def list_creator_accounts(db: Session = DbDep, user: User = CurrentUserDep):
    stmt = select(CreatorAccount).where(CreatorAccount.user_id == user.id)
    return list(db.scalars(stmt))


@router.post("", response_model=CreatorAccountOut, status_code=201)
def create_creator_account(
    body: CreatorAccountCreate, db: Session = DbDep, user: User = CurrentUserDep
) -> CreatorAccount:
    """Manual account registration -- see CreatorAccountCreate's docstring.
    Still useful even with real TikTok OAuth now wired up below: for
    platforms that don't have OAuth here yet (youtube, instagram), or for
    registering a token from TikTok's own sandbox tooling without a full
    consent-screen round trip. Encrypted before it ever reaches the DB.
    Reused by upload.py exactly like an OAuth-populated row would be --
    nothing downstream needs to know which path created it.
    """
    daily_cap = body.daily_upload_cap if body.daily_upload_cap is not None else settings.default_daily_upload_cap
    if not (1 <= daily_cap <= settings.hard_max_daily_upload_cap):
        raise ApiError(
            400, "invalid_daily_upload_cap",
            f"daily_upload_cap must be between 1 and {settings.hard_max_daily_upload_cap}",
        )

    account = CreatorAccount(
        user_id=user.id,
        platform=body.platform,
        external_account_id=body.external_account_id,
        access_token_encrypted=encrypt_token(body.access_token),
        refresh_token_encrypted=encrypt_token(body.refresh_token) if body.refresh_token else None,
        daily_upload_cap=daily_cap,
    )
    db.add(account)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ApiError(
            409, "creator_account_already_exists",
            "An account for this platform + external_account_id already exists",
        ) from exc
    db.refresh(account)
    return account


@router.get("/tiktok/oauth/start", response_model=TikTokOAuthStartOut)
def tiktok_oauth_start(user: User = CurrentUserDep) -> dict:
    """Returns the URL to send the creator's browser to for TikTok's
    consent screen. A normal authenticated JSON call (needs CurrentUserDep
    to know who's connecting) -- the caller does the actual
    `window.location = authorize_url` redirect itself; this endpoint
    doesn't redirect server-side.
    """
    if not settings.tiktok_client_key or not settings.tiktok_redirect_uri:
        raise ApiError(
            503, "tiktok_not_configured",
            "TIKTOK_CLIENT_KEY and TIKTOK_REDIRECT_URI must both be set before connecting a TikTok account "
            "-- see README's TikTok setup section.",
        )
    state = create_oauth_state_token(str(user.id))
    return {"authorize_url": tiktok_client.build_authorize_url(redirect_uri=settings.tiktok_redirect_uri, state=state)}


@router.get("/tiktok/oauth/callback", response_class=HTMLResponse)
def tiktok_oauth_callback(
    db: Session = DbDep,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    error_description: str | None = Query(None),
):
    """Where TikTok's redirect actually lands after the creator approves
    (or denies) access on the consent screen -- a plain browser GET
    navigation, not an API call from our own frontend, so there's no
    bearer header to authenticate with and no JSON caller to hand a
    {error:{...}} envelope back to. Returns a small standalone HTML page
    instead, either way.
    """
    if error:
        return _oauth_result_page(ok=False, message=f"TikTok declined the connection: {error_description or error}")
    if not code or not state:
        return _oauth_result_page(ok=False, message="TikTok's redirect was missing code/state -- try connecting again.")

    try:
        user_id = decode_oauth_state_token(state)
    except Exception:
        log.warning("tiktok_oauth.invalid_state")
        return _oauth_result_page(
            ok=False, message="This connection link expired or is invalid -- go back and try connecting again."
        )

    try:
        token_data = tiktok_client.exchange_code_for_token(code, settings.tiktok_redirect_uri)
    except tiktok_client.TikTokAPIError as exc:
        log.error("tiktok_oauth.token_exchange_failed", error=str(exc))
        return _oauth_result_page(ok=False, message=f"TikTok rejected the connection: {exc}")

    access_token = token_data.get("access_token")
    open_id = token_data.get("open_id")
    if not access_token or not open_id:
        log.error("tiktok_oauth.incomplete_token_response", keys=list(token_data.keys()))
        return _oauth_result_page(ok=False, message="TikTok's response was missing required fields -- try again.")

    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in")
    scope = token_data.get("scope") or ""
    token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in)) if expires_in else None

    # Upsert rather than always-insert: re-connecting an already-registered
    # account (token refresh, re-granting after revocation) should update
    # the stored tokens in place, not 409 the way a duplicate manual
    # registration does in create_creator_account above.
    existing = db.scalars(
        select(CreatorAccount).where(
            CreatorAccount.user_id == uuid.UUID(user_id),
            CreatorAccount.platform == "tiktok",
            CreatorAccount.external_account_id == open_id,
        )
    ).one_or_none()

    if existing is not None:
        existing.access_token_encrypted = encrypt_token(access_token)
        if refresh_token:
            existing.refresh_token_encrypted = encrypt_token(refresh_token)
        existing.token_expires_at = token_expires_at
        if scope:
            existing.scopes = scope.split(",")
    else:
        db.add(
            CreatorAccount(
                user_id=uuid.UUID(user_id),
                platform="tiktok",
                external_account_id=open_id,
                access_token_encrypted=encrypt_token(access_token),
                refresh_token_encrypted=encrypt_token(refresh_token) if refresh_token else None,
                token_expires_at=token_expires_at,
                scopes=scope.split(",") if scope else [],
                daily_upload_cap=settings.default_daily_upload_cap,
            )
        )
    db.commit()
    log.info("tiktok_oauth.connected", open_id=open_id)

    return _oauth_result_page(ok=True, message="TikTok account connected. You can close this tab.")


def _oauth_result_page(*, ok: bool, message: str) -> HTMLResponse:
    # Deliberately minimal -- this is a redirect landing page, not part of
    # the dev console's own UI (web/); a creator sees this for a couple of
    # seconds after approving/declining on TikTok's own consent screen.
    color = "#3fbf76" if ok else "#e05a5a"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Clipping Machine</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background:#0f1115; color:#e6e8eb; display:flex; align-items:center;
          justify-content:center; height:100vh; margin:0; }}
  .card {{ text-align:center; max-width:420px; padding:2rem; }}
  .status {{ color:{color}; font-size:1.1rem; font-weight:600; margin-bottom:0.5rem; }}
</style></head>
<body><div class="card"><div class="status">{"Connected" if ok else "Connection failed"}</div>
<p>{message}</p></div></body></html>"""
    return HTMLResponse(content=html, status_code=200 if ok else 400)
