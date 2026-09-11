"""Create a local dev user and print a bearer token for it.

Usage:
    python scripts/create_dev_user.py you@example.com

There's no signup/login flow in the MVP scaffold (see app/core/auth.py) --
this is the dev-only stand-in until one exists.
"""
from __future__ import annotations

import sys

from app.core.auth import create_access_token
from app.db.models import User
from app.db.session import SessionLocal


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/create_dev_user.py <email>")
    email = sys.argv[1]

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is None:
            user = User(email=email, password_hash="dev-user-no-password", display_name=email.split("@")[0])
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created user {user.id} ({email})")
        else:
            print(f"Found existing user {user.id} ({email})")

        token = create_access_token(str(user.id))
        print(f"\nBearer token:\n{token}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
