"""Which LLM a given creator's jobs should use.

Both LLM call sites (caption/title generation and segment suggestion) used
to read app.core.config.settings directly, which made the provider a
deployment-wide choice. A creator supplying their own OpenAI key needs it to
be a per-account one instead.

Resolution order, per field:
  1. the user's own setting, when they have one
  2. settings.* -- the .env default, i.e. exactly the old behaviour

So an account that has configured nothing behaves precisely as before, and
this module is the only place that knows the difference.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.core.crypto import decrypt_token
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LlmConfig:
    """Everything a call site needs to build an OpenAI-compatible client.

    `api_key` is a real secret -- never log this object, and never return it
    from an API route. __repr__ is overridden below so an accidental log line
    or traceback cannot leak it.
    """

    provider: str
    api_key: str
    model: str
    base_url: str | None

    @property
    def usable(self) -> bool:
        """False when the provider is openai but no key is available -- the
        caller should fall back to its heuristic rather than making a call
        that is certain to 401."""
        return bool(self.api_key) and not (self.provider == "openai" and not self.api_key)

    def __repr__(self) -> str:
        shown = f"...{self.api_key[-4:]}" if len(self.api_key) > 4 else "set" if self.api_key else "unset"
        return f"LlmConfig(provider={self.provider!r}, model={self.model!r}, api_key={shown})"


def user_openai_key(user) -> str:
    """The user's own decrypted key, or "" if they have none.

    A key that fails to decrypt (rotated CREATOR_TOKEN_ENCRYPTION_KEY, or a
    hand-edited row) is treated as absent rather than raising: the caller
    falls back to the .env key or to its heuristic, which is a far better
    outcome than failing every job on the account.
    """
    encrypted = getattr(user, "openai_api_key_encrypted", None)
    if not encrypted:
        return ""
    try:
        return decrypt_token(encrypted)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning("llm_config.key_decrypt_failed", error=str(exc))
        return ""


def resolve_llm_config(user=None) -> LlmConfig:
    """Build the effective LLM configuration for this user's jobs."""
    provider = (getattr(user, "llm_provider", None) or settings.caption_llm_provider or "ollama").lower()

    if provider == "ollama":
        return LlmConfig(
            provider="ollama",
            # Ollama ignores the key entirely, but the openai client refuses
            # to construct without a non-empty string.
            api_key="ollama",
            model=settings.caption_ollama_model,
            base_url=settings.caption_ollama_base_url,
        )

    # openai: prefer the creator's own key, fall back to the deployment's
    return LlmConfig(
        provider="openai",
        api_key=user_openai_key(user) or settings.openai_api_key,
        model=settings.caption_llm_model,
        base_url=None,
    )


def resolve_llm_config_for_user_id(user_id) -> LlmConfig:
    """resolve_llm_config, loading the user in its own short-lived session.

    For workers, which have a user_id rather than a User object. Any failure
    degrades to the .env defaults instead of failing the job.
    """
    if user_id is None:
        return resolve_llm_config(None)
    try:
        import uuid as _uuid

        from app.db.models import User
        from app.workers.common import db_session

        uid = user_id if isinstance(user_id, _uuid.UUID) else _uuid.UUID(str(user_id))
        with db_session() as db:
            return resolve_llm_config(db.get(User, uid))
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm_config.user_lookup_failed", error=str(exc))
        return resolve_llm_config(None)
