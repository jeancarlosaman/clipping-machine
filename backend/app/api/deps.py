"""FastAPI dependencies shared across routers."""
from __future__ import annotations

import uuid
from collections.abc import Generator

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.api.errors import ApiError
from app.core.auth import decode_access_token
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


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = BearerDep,
    db: Session = DbDep,
) -> User:
    if credentials is None:
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

    return user


CurrentUserDep = Depends(get_current_user)


def get_session() -> Generator[Session, None, None]:
    yield from get_db()
