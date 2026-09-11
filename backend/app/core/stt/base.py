"""Shared types for STT providers.

Every provider (local Whisper, OpenAI's hosted API, ...) implements the
same `transcribe(wav_path) -> SttResult` shape so app/workers/transcription.py
never branches on which one is active -- see app/core/stt/__init__.py's
get_stt_provider() for the selection point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SttSegment:
    start: float
    end: float
    text: str


@dataclass
class SttResult:
    segments: list[SttSegment] = field(default_factory=list)
    full_text: str = ""
    language: str | None = None


class SttProvider(Protocol):
    name: str

    def transcribe(self, wav_path: str) -> SttResult:
        ...


class TransientSttError(Exception):
    """Retry-worthy: network error, rate limit, provider 5xx/timeout."""


class PermanentSttError(Exception):
    """Not retry-worthy: bad/unsupported audio, missing config, provider 4xx."""
