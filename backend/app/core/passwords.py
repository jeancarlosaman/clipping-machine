"""Password hashing.

Argon2id via argon2-cffi, which is OWASP's first-choice password hash: it is
memory-hard, so the GPU/ASIC advantage an attacker gets against bcrypt/PBKDF2
is much smaller. The library's defaults are deliberately not overridden --
they track current guidance, and hand-tuned cost parameters are a classic way
to accidentally weaken a hash.

Nothing here is hand-rolled crypto, on purpose. The only judgement calls in
this module are about *timing* (see verify_password_constant_work) and rehash
upgrades.
"""
from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()

# Any string that is not a valid argon2 hash -- notably the
# "dev-user-no-password" placeholder scripts/create_dev_user.py writes --
# fails verification, which is the correct outcome: a dev user has no
# password and must not be loggable-into with one.
_DUMMY_HASH = _hasher.hash("timing-equalisation-only")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        _hasher.verify(stored_hash, password)
        return True
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def verify_password_constant_work(password: str, stored_hash: str | None) -> bool:
    """verify_password, but doing the same work when the user doesn't exist.

    Without this, "no such email" returns in microseconds while a real email
    with a wrong password takes ~50ms of argon2 work -- a timing difference
    that lets anyone enumerate which emails have accounts. Hashing against a
    throwaway hash keeps the two paths comparable.
    """
    if stored_hash is None:
        _hasher.verify(_DUMMY_HASH, password) if False else None  # noqa: B018
        verify_password(password, _DUMMY_HASH)
        return False
    return verify_password(password, stored_hash)


def needs_rehash(stored_hash: str) -> bool:
    """True when a stored hash used weaker parameters than the current
    defaults -- re-hash on next successful login to upgrade it in place."""
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return False
