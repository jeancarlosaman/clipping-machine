"""Minimal JWT bearer auth.

MVP assumption (see architecture doc §6): every API endpoint requires a
JWT bearer token; `user_id` is read from the token's `sub` claim and never
trusted from the request body. This module only does encode/decode -- it
does not implement signup/login/password handling, which is out of scope
for the pipeline MVP and should be a small, separate piece of work before
this goes anywhere near real users.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings

ALGORITHM = "HS256"


_ACCESS_TYP = "access"


def create_access_token(user_id: str, expires_in_minutes: int = 60 * 24) -> str:
    """Mint a token for `user_id`. Used by tests/local dev; a real login
    endpoint would call this after verifying credentials."""
    payload = {
        "sub": user_id,
        "typ": _ACCESS_TYP,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """Return the user_id (sub claim) from a valid token, or raise
    jwt.PyJWTError. Also rejects (ValueError) a token minted for a
    different purpose but signed with the same jwt_secret -- concretely,
    an OAuth `state` token (see create_oauth_state_token below) must never
    work as a bearer access token just because both happen to be
    HS256-signed with the same key. A missing `typ` claim is accepted as
    `_ACCESS_TYP` for backward compatibility with tokens minted before this
    check existed -- everything issued by this codebase's own
    create_access_token now sets it explicitly."""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    typ = payload.get("typ", _ACCESS_TYP)
    if typ != _ACCESS_TYP:
        raise ValueError(f"token typ={typ!r} is not a valid access token")
    return payload["sub"]


# --- TikTok OAuth `state` param (app.api.routers.creator_accounts) ---
#
# TikTok's redirect lands the user's browser directly on our callback route
# with `?code&state`, unauthenticated (no bearer header -- it's a plain
# browser navigation, not an API call from our own frontend). The callback
# still needs to know which of *our* users is connecting an account. Rather
# than trust a client-supplied user_id (forgeable) or require a login at
# the callback (loses the OAuth round trip's context), the "start" endpoint
# mints one of these short-lived tokens and passes it as `state`; the
# callback decodes it to recover user_id. `typ` distinguishes this from a
# real bearer access token -- both are HS256-signed with the same
# jwt_secret, so without that check a stolen/leaked state token would be a
# valid (if short-lived) access token for the account it names, which is
# not a trade a CSRF-protection value should be able to make.
_OAUTH_STATE_TYP = "tiktok_oauth_state"


def create_oauth_state_token(user_id: str, expires_in_minutes: int = 15) -> str:
    """Mint the `state` value for a TikTok OAuth authorize redirect. Short
    expiry -- this only needs to survive the user's own consent-screen
    round trip, not sit around like a real session token."""
    payload = {
        "sub": user_id,
        "typ": _OAUTH_STATE_TYP,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_oauth_state_token(token: str) -> str:
    """Return the user_id embedded by create_oauth_state_token. Raises
    jwt.PyJWTError for an expired/invalid/missing token, or ValueError if
    the signature is valid but `typ` doesn't match -- e.g. someone passed a
    normal access token (or nothing at all, if TikTok's redirect dropped
    `state`) as the OAuth `state` param."""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    if payload.get("typ") != _OAUTH_STATE_TYP:
        raise ValueError("token is not a TikTok OAuth state token")
    return payload["sub"]
