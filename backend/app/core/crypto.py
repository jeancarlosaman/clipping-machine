"""Symmetric encryption for creator account tokens at rest.

`creator_accounts.access_token_encrypted` / `refresh_token_encrypted` are
real OAuth-style credentials (or, until real TikTok OAuth is wired up, a
manually-pasted token -- see app/api/routers/creator_accounts.py) -- they
shouldn't sit in the database as plaintext even in dev, since a DB dump/leak
would otherwise hand over a live account credential directly. Fernet
(symmetric, authenticated encryption from the `cryptography` package) is
enough here: this is at-rest protection for one column, not a general
crypto system, and Fernet's built-in key rotation via MultiFernet is the
natural next step if CREATOR_ACCOUNT_ENCRYPTION_KEY ever needs rotating.
"""
from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def _fernet() -> Fernet:
    key_material = settings.creator_account_encryption_key
    if key_material:
        # Must already be a valid urlsafe-base64-encoded 32-byte key (what
        # Fernet.generate_key() produces) -- fail loudly at first use rather
        # than silently encrypting with something Fernet will also reject
        # later, which would be a much more confusing failure to debug.
        return Fernet(key_material.encode("utf-8"))
    # Dev convenience: derive a stable key from jwt_secret so this works
    # without extra setup locally. NOT a substitute for setting a real,
    # dedicated CREATOR_ACCOUNT_ENCRYPTION_KEY outside of local dev --
    # rotating jwt_secret would otherwise also silently break decrypting
    # every already-stored token.
    derived = hashlib.sha256(settings.jwt_secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_token(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_token(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        # Wrong/rotated key, or the value predates encryption being added --
        # surfaced as a clear error rather than a cryptic Fernet exception,
        # since this can also legitimately happen after CREATOR_ACCOUNT_ENCRYPTION_KEY
        # changes and is worth distinguishing from "the token itself is bad."
        raise ValueError("Could not decrypt stored token -- wrong/rotated encryption key?") from exc
