"""Hosted STT via the OpenAI Whisper API.

Opt-in provider (STT_PROVIDER=openai, requires OPENAI_API_KEY) -- no local
compute cost, scales without this machine's hardware mattering, but has a
per-minute API cost and a network dependency. See app/core/stt/local_whisper.py
for the open-source default.

Chunks audio before calling the API (app/core/stt/chunking.py) since the
endpoint enforces a hard request-size limit that most VOD-length audio
exceeds. Each chunk is transcribed independently and its segment timestamps
are offset back into the original audio's timeline before merging.
"""
from __future__ import annotations

import json
import tempfile

from app.core.config import settings
from app.workers.common import logger, run_subprocess

from .base import PermanentSttError, SttResult, SttSegment, TransientSttError
from .chunking import plan_chunk_boundaries, split_audio


def _probe_duration_seconds(wav_path: str) -> float:
    result = run_subprocess(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", wav_path]
    )
    return float(json.loads(result.stdout)["format"]["duration"])


class OpenAiSttProvider:
    name = "openai"

    def __init__(self):
        if not settings.openai_api_key:
            raise PermanentSttError("STT_PROVIDER=openai but OPENAI_API_KEY is not set")
        import openai  # local import: keep the dependency optional for local-only dev

        self._openai = openai
        self._client = openai.OpenAI(api_key=settings.openai_api_key)

    def transcribe(self, wav_path: str) -> SttResult:
        duration = _probe_duration_seconds(wav_path)
        boundaries = plan_chunk_boundaries(
            wav_path, duration, target_chunk_seconds=settings.stt_openai_chunk_seconds
        )

        if not boundaries:
            return self._transcribe_one(wav_path, offset_seconds=0.0)

        logger.info("stt.openai.chunking", chunk_count=len(boundaries) + 1, duration=duration)
        with tempfile.TemporaryDirectory(prefix="stt-openai-chunks-") as tmp_dir:
            chunks = split_audio(wav_path, boundaries, tmp_dir)
            results = [self._transcribe_one(c.path, c.offset_seconds) for c in chunks]

        segments = [seg for r in results for seg in r.segments]
        full_text = " ".join(r.full_text for r in results if r.full_text).strip()
        language = next((r.language for r in results if r.language), None)
        return SttResult(segments=segments, full_text=full_text, language=language)

    def _transcribe_one(self, path: str, offset_seconds: float) -> SttResult:
        try:
            with open(path, "rb") as f:
                response = self._client.audio.transcriptions.create(
                    model=settings.stt_openai_model,
                    file=f,
                    response_format="verbose_json",
                )
        except self._openai.RateLimitError as exc:
            raise TransientSttError(f"rate limited: {exc}") from exc
        except self._openai.APIConnectionError as exc:
            raise TransientSttError(f"connection error: {exc}") from exc
        except self._openai.APIStatusError as exc:
            if 500 <= exc.status_code < 600:
                raise TransientSttError(f"OpenAI {exc.status_code}: {exc}") from exc
            raise PermanentSttError(f"OpenAI {exc.status_code}: {exc}") from exc

        raw_segments = getattr(response, "segments", None) or []
        segments = [
            SttSegment(
                start=float(_seg_field(s, "start")) + offset_seconds,
                end=float(_seg_field(s, "end")) + offset_seconds,
                text=str(_seg_field(s, "text")).strip(),
            )
            for s in raw_segments
        ]
        full_text = str(getattr(response, "text", "") or "").strip()
        language = getattr(response, "language", None)
        return SttResult(segments=segments, full_text=full_text, language=language)


def _seg_field(segment, field: str):
    """The openai SDK returns pydantic objects for verbose_json segments in
    most versions, but tests (and possibly future SDK versions) may pass
    plain dicts -- support both rather than pinning to one shape."""
    if isinstance(segment, dict):
        return segment[field]
    return getattr(segment, field)
