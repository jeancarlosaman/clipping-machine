"""Per-account settings -- currently the LLM provider and API key.

Why this exists: which model reads your transcript used to be a
deployment-wide .env choice, so switching from the local Ollama model to a
hosted one meant editing a file and restarting the API. That is the single
biggest lever on clip-selection quality, so it belongs in the UI.

Security posture for the API key, which can spend the holder's money:
  * stored encrypted (app.core.crypto, the same Fernet helper protecting
    creator_accounts' TikTok tokens) -- never plaintext in the database
  * never returned by any route; GET reports only whether one is set and
    its last four characters, which is enough to tell two keys apart
  * cleared by sending an explicit null, not by sending an empty string,
    so a blank form field can never wipe a working key by accident
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUserDep, DbDep
from app.api.errors import ApiError
from app.core.config import settings
from app.core.crypto import encrypt_token
from app.core.llm_config import resolve_llm_config, user_openai_key
from app.db.models import User

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

LLM_PROVIDERS = ("openai", "ollama")


class LlmSettingsOut(BaseModel):
    # What this account chose; null means "follow the server default".
    provider: str | None
    # What will actually be used, after falling back to the .env default.
    effective_provider: str
    has_own_key: bool
    key_hint: str | None
    # Read-only context, so the UI can explain what each choice will do
    # rather than making the creator guess.
    openai_model: str
    ollama_model: str
    ollama_base_url: str
    server_has_key: bool
    segment_suggestions_enabled: bool


class LlmSettingsIn(BaseModel):
    provider: str | None = None
    # Absent = leave the stored key alone. Explicit null = delete it.
    # A string = replace it.
    api_key: str | None = Field(default=None)


def _to_out(user: User) -> LlmSettingsOut:
    key = user_openai_key(user)
    return LlmSettingsOut(
        provider=user.llm_provider,
        effective_provider=resolve_llm_config(user).provider,
        has_own_key=bool(key),
        key_hint=f"…{key[-4:]}" if len(key) >= 4 else None,
        openai_model=settings.caption_llm_model,
        ollama_model=settings.caption_ollama_model,
        ollama_base_url=settings.caption_ollama_base_url,
        server_has_key=bool(settings.openai_api_key),
        segment_suggestions_enabled=settings.enable_llm_segment_suggestions,
    )


@router.get("/llm", response_model=LlmSettingsOut)
def get_llm_settings(user: User = CurrentUserDep) -> LlmSettingsOut:
    return _to_out(user)


@router.put("/llm", response_model=LlmSettingsOut)
def update_llm_settings(
    payload: LlmSettingsIn,
    db: Session = DbDep,
    user: User = CurrentUserDep,
) -> LlmSettingsOut:
    provided = payload.model_fields_set

    if "provider" in provided:
        if payload.provider is not None and payload.provider not in LLM_PROVIDERS:
            raise ApiError(400, "invalid_provider", f"provider must be one of {list(LLM_PROVIDERS)} or null")
        user.llm_provider = payload.provider

    if "api_key" in provided:
        if payload.api_key is None:
            user.openai_api_key_encrypted = None
        else:
            key = payload.api_key.strip()
            if not key:
                raise ApiError(
                    400, "invalid_api_key",
                    "Send null to remove the key; an empty string is rejected so a blank "
                    "form field cannot silently delete a working key.",
                )
            # Not a format check on the vendor's prefix -- those change, and
            # rejecting a valid future key is worse than letting a bad one
            # through and failing at call time with a clear provider error.
            if len(key) < 20:
                raise ApiError(400, "invalid_api_key", "That does not look like an API key (too short).")
            user.openai_api_key_encrypted = encrypt_token(key)

    # Choosing openai with no key anywhere would silently fall back to the
    # heuristic captions -- better to say so now than after a render.
    effective = user.llm_provider or settings.caption_llm_provider
    if effective == "openai" and not (user_openai_key(user) or settings.openai_api_key):
        raise ApiError(
            400, "no_api_key",
            "Select OpenAI only with an API key saved -- otherwise captions and clip "
            "suggestions silently fall back to the non-LLM heuristics.",
        )

    db.commit()
    db.refresh(user)
    return _to_out(user)
