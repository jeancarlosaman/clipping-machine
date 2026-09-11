"""Speech-to-text provider selection.

STT_PROVIDER in config ("local" | "openai") picks the implementation --
app/workers/transcription.py only ever talks to the SttProvider interface,
never imports a concrete provider directly, so swapping providers (or
adding a third) never touches worker code.

Concrete providers are imported lazily inside get_stt_provider(), not at
module load time, so an unused provider's dependency (faster-whisper's
ctranslate2/onnxruntime stack is not tiny) doesn't get imported into every
process that merely imports this package.
"""
from __future__ import annotations

from app.core.config import settings

from .base import PermanentSttError, SttProvider, SttResult, SttSegment, TransientSttError


def get_stt_provider(model_size: str | None = None) -> SttProvider:
    """`model_size` is a per-job override (see StreamJob.stt_model_size) for
    the local provider's faster-whisper model size -- None uses
    settings.stt_local_model_size as before. Ignored for the openai provider,
    which has its own, unrelated model setting (STT_OPENAI_MODEL); accepting
    (and ignoring) the parameter there too keeps this a single call site in
    transcription.py that never needs to branch on which provider is active.
    """
    if settings.stt_provider == "openai":
        from .openai_api import OpenAiSttProvider

        return OpenAiSttProvider()
    if settings.stt_provider == "local":
        from .local_whisper import LocalWhisperProvider

        return LocalWhisperProvider(model_size=model_size)
    raise ValueError(f"Unknown STT_PROVIDER: {settings.stt_provider!r} (expected 'local' or 'openai')")


__all__ = [
    "get_stt_provider",
    "SttProvider",
    "SttResult",
    "SttSegment",
    "PermanentSttError",
    "TransientSttError",
]
