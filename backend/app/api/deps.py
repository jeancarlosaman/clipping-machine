"""FastAPI dependencies shared across routers."""
from __future__ import annotations

import uuid
from collections.abc import Generator

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.api.errors import ApiError
from app.core.auth import decode_access_token
from app.core.config import settings
from app.db.models import User
from app.db.session import get_db

DbDep = Depends(get_db)

# HTTPBearer (rather than reading the Authorization header by hand) is what
# registers a "bearerAuth" security scheme in the OpenAPI schema -- that's
# the piece that makes Swagger UI show an "Authorize" button at all.
# auto_error=False so a missing header becomes our own ApiError(401, ...)
# with the standard {error:{...}} envelope, instead of FastAPI's default
# {"detail": "Not authenticated"} shape.
_bearer_scheme = HTTPBearer(auto_error=False)
BearerDep = Depends(_bearer_scheme)


def _dev_auto_login_user(db: Session) -> User | None:
    """The configured local-dev user, or None if auto-login is not active.

    Active only when BOTH settings.app_env == "development" AND
    settings.dev_auto_login_email is set -- see that setting's comment for
    why it takes two conditions rather than one. Returning None here means
    the caller 401s exactly as it did before this existed, so production
    behaviour is unchanged.
    """
    if settings.app_env != "development" or not settings.dev_auto_login_email:
        return None

    email = settings.dev_auto_login_email
    user = db.query(User).filter(User.email == email).one_or_none()
    if user is None:
        # Same shape scripts/create_dev_user.py produces -- no password, since
        # there is no login flow to use one with.
        user = User(email=email, password_hash="dev-user-no-password", display_name=email.split("@")[0])
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = BearerDep,
    db: Session = DbDep,
) -> User:
    # Two ways in, checked in this order:
    #   1. Authorization: Bearer <jwt>  -- scripts, curl, the README examples
    #   2. the session cookie           -- the browser console after login
    # The cookie holds the same JWT, so everything below this point is
    # identical either way. An explicit Authorization header wins, so a
    # script run in a browser-authenticated context still acts as whoever
    # the header says.
    token_from_cookie = request.cookies.get(settings.session_cookie_name)

    if credentials is None and token_from_cookie:
        try:
            user_id = decode_access_token(token_from_cookie)
            user = db.get(User, uuid.UUID(user_id))
            if user is not None:
                request.state.auth_source = "session"
                return user
        except (jwt.PyJWTError, ValueError):
            # A stale or tampered cookie falls through to the normal
            # unauthenticated path rather than 500ing -- the browser just
            # gets a 401 and the console sends it to the login page.
            pass

    if credentials is None:
        dev_user = _dev_auto_login_user(db)
        if dev_user is not None:
            # Recorded so the UI can tell a real session from a dev bypass.
            # Without this, signing out bounces straight back in: the cookie
            # is cleared but dev auto-login answers /me anyway, and the login
            # page redirects to a console it thinks you are signed into.
            request.state.auth_source = "dev_auto_login"
            return dev_user
        raise ApiError(401, "unauthenticated", "Missing or malformed Authorization header")

    token = credentials.credentials
    try:
        user_id = decode_access_token(token)
    except (jwt.PyJWTError, ValueError) as exc:
        # ValueError covers decode_access_token's own typ check -- a
        # well-signed token minted for a different purpose (e.g. a TikTok
        # OAuth state token, see app.core.auth) must 401 here just like a
        # bad signature would, not 500.
        raise ApiError(401, "unauthenticated", "Invalid or expired token") from exc

    try:
        user = db.get(User, uuid.UUID(user_id))
    except ValueError as exc:
        raise ApiError(401, "unauthenticated", "Token subject is not a valid user id") from exc

    if user is None:
        raise ApiError(401, "unauthenticated", "Token does not correspond to a known user")

    request.state.auth_source = "bearer"
    return user


CurrentUserDep = Depends(get_current_user)


def get_session() -> Generator[Session, None, None]:
    yield from get_db()
