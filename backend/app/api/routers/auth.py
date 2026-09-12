"""Signup / login / logout -- architecture doc §6.

Replaces the dev-only "paste a bearer token you minted from a CLI script"
flow. The token itself is unchanged (same JWT, same `sub` claim), so
scripts/create_dev_user.py, the README's curl examples and any API client
keep working exactly as before; this module just adds a way to *obtain* one
with an email and password, and a browser session cookie so the console
never has to hold the token in JavaScript.

Two deliberate choices worth not "simplifying" later:

1. The session rides in an httpOnly cookie, not localStorage. JavaScript
   cannot read an httpOnly cookie, so an XSS bug in the console cannot walk
   off with a session that is able to post to the creator's TikTok (see
   creator_accounts.access_token_encrypted). SameSite=Lax is what stops a
   third-party page from using that cookie for a cross-site POST; every
   state-changing route here is POST/PUT/DELETE, so Lax covers them.

2. Login failures are deliberately indistinguishable. Wrong password and
   unknown email return the same message and do the same amount of hashing
   work (see passwords.verify_password_constant_work) -- otherwise the API
   is an oracle for "does this person have an account".

NOT implemented, on purpose: password reset. It needs an email provider and
it is the single most commonly broken auth flow there is; a half-built one
is worse than none. Until it exists, a forgotten password is recovered by
running scripts/create_dev_user.py or updating the row directly.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUserDep, DbDep
from app.api.errors import ApiError
from app.core.auth import create_access_token
from app.core.config import settings
from app.core.logging import get_logger
from app.core.passwords import (
    hash_password,
    needs_rehash,
    verify_password_constant_work,
)
from app.db.models import User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = get_logger(__name__)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)
    display_name: str | None = None
    invite_code: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str | None
    # Only populated by /me: "session" (cookie), "bearer" (Authorization
    # header) or "dev_auto_login" (no credential at all -- see
    # settings.dev_auto_login_email). The console needs the distinction
    # because sign-out is meaningless in the last case.
    auth_source: str | None = None

    model_config = {"from_attributes": True}


def _normalize_email(email: str) -> str:
    """Lower-cased and trimmed. Without this, Bob@x.com and bob@x.com are two
    accounts, and the unique constraint on users.email won't stop it."""
    return email.strip().lower()


def _set_session_cookie(response: Response, user_id: str) -> None:
    token = create_access_token(user_id, expires_in_minutes=settings.session_ttl_minutes)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=settings.session_ttl_minutes * 60,
        httponly=True,
        # Secure would make the cookie invisible over plain http, which is
        # exactly how local dev runs -- so it follows the environment rather
        # than being hardcoded either way.
        secure=settings.app_env != "development",
        samesite="lax",
        path="/",
    )


def _rate_limit_key(request: Request, email: str) -> str:
    client = request.client.host if request.client else "unknown"
    return f"login-attempts:{client}:{email}"


def _check_and_count_attempt(request: Request, email: str) -> None:
    """Throttle failed logins per email+IP using the Redis that already backs
    the job queue.

    Never raises on a Redis problem: the queue being down should stop jobs,
    not stop the creator logging in to look at why. Losing throttling for the
    duration of a Redis outage is the lesser failure.
    """
    try:
        from app.core.queue import redis_connection

        conn = redis_connection()
        key = _rate_limit_key(request, email)
        attempts = conn.incr(key)
        if attempts == 1:
            conn.expire(key, settings.login_attempt_window_seconds)
        if attempts > settings.login_max_attempts:
            raise ApiError(
                429, "too_many_attempts",
                "Too many failed sign-in attempts. Wait a few minutes and try again.",
            )
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning("auth.rate_limit_unavailable", error=str(exc))


def _clear_attempts(request: Request, email: str) -> None:
    try:
        from app.core.queue import redis_connection

        redis_connection().delete(_rate_limit_key(request, email))
    except Exception:  # noqa: BLE001
        pass


@router.post("/signup", response_model=UserOut, status_code=201)
def signup(payload: SignupRequest, response: Response, db: Session = DbDep) -> UserOut:
    if settings.signup_invite_code:
        if (payload.invite_code or "") != settings.signup_invite_code:
            raise ApiError(403, "invite_required", "A valid invite code is required to sign up.")

    if len(payload.password) < settings.min_password_length:
        raise ApiError(
            400, "weak_password",
            f"Password must be at least {settings.min_password_length} characters.",
        )

    email = _normalize_email(payload.email)
    if db.query(User).filter(User.email == email).one_or_none() is not None:
        # Signup genuinely cannot hide that an email is taken -- it has to
        # refuse the duplicate. (Login, which can hide it, does.)
        raise ApiError(409, "email_taken", "An account with that email already exists.")

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=(payload.display_name or "").strip() or email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    _set_session_cookie(response, str(user.id))
    logger.info("auth.signup", user_id=str(user.id))
    return UserOut(id=str(user.id), email=user.email, display_name=user.display_name)


@router.post("/login", response_model=UserOut)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = DbDep) -> UserOut:
    email = _normalize_email(payload.email)
    _check_and_count_attempt(request, email)

    user = db.query(User).filter(User.email == email).one_or_none()
    stored = user.password_hash if user else None

    if not verify_password_constant_work(payload.password, stored):
        # Same message for "no such account" and "wrong password" -- see the
        # module docstring.
        raise ApiError(401, "invalid_credentials", "Incorrect email or password.")

    # Upgrade the stored hash in place if argon2's defaults have moved on
    # since this password was set.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        db.commit()

    _clear_attempts(request, email)
    _set_session_cookie(response, str(user.id))
    logger.info("auth.login", user_id=str(user.id))
    return UserOut(id=str(user.id), email=user.email, display_name=user.display_name)


# response_model=None is load-bearing, and matches the other 204 routes in
# this API (clips.delete_clip, stream_jobs.delete_stream_job). Without it
# FastAPI reads the `-> None` annotation as a response model -- NoneType is
# truthy -- then asserts that a 204 must not have a body, and the entire app
# fails to import.
@router.post("/logout", status_code=204, response_model=None)
def logout(response: Response) -> None:
    # delete_cookie has to match the path the cookie was set with, or the
    # browser keeps it.
    response.delete_cookie(settings.session_cookie_name, path="/")


@router.get("/me", response_model=UserOut)
def me(request: Request, user: User = CurrentUserDep) -> UserOut:
    """Who the current session belongs to, and how they got here.

    The console calls this on load to decide whether to show the app or
    redirect to the login page; the login page calls it to skip the form if
    you are already signed in. `auth_source` is what stops those two
    bouncing off each other when dev auto-login is enabled.
    """
    return UserOut(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        auth_source=getattr(request.state, "auth_source", None),
    )
