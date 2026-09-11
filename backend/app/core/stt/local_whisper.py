"""Local, open-source STT via faster-whisper (CTranslate2 port of Whisper).

Default provider (STT_PROVIDER=local) -- no per-minute cost, no API key,
audio never leaves the machine. Runs on CPU by default (see
STT_LOCAL_MODEL_SIZE / STT_LOCAL_COMPUTE_TYPE in .env.example for the
speed/accuracy trade-off knobs); pass device="cuda" below if a GPU is ever
available to this worker -- not wired up as a setting yet since the
project's current deployment target is CPU-only.

No manual chunking here (contrast app/core/stt/openai_api.py): faster-whisper
streams the whole file itself and handles files far longer than a VOD would
ever be. `vad_filter=True` also gets us silence-aware segmentation for free,
which is the same property the OpenAI path has to build by hand via
app/core/stt/chunking.py.
"""
from __future__ import annotations

from app.core.config import settings
from app.workers.common import logger

from .base import PermanentSttError, SttResult, SttSegment

_model_cache: dict[str, object] = {}


def _get_model(model_size: str):
    # Cache key includes model_size so a long-running worker process that
    # transcribes jobs with different per-job stt_model_size overrides (see
    # StreamJob.stt_model_size) keeps every size it's loaded so far resident
    # rather than reloading from disk on every job -- trades some memory for
    # not re-paying model load time each time the size changes.
    key = f"{model_size}:{settings.stt_local_compute_type}"
    if key not in _model_cache:
        from faster_whisper import WhisperModel  # local import: heavy, only load if actually used

        logger.info("stt.local.loading_model", model_size=model_size)
        _model_cache[key] = WhisperModel(
            model_size,
            device="cpu",
            compute_type=settings.stt_local_compute_type,
        )
    return _model_cache[key]


class LocalWhisperProvider:
    name = "local-whisper"

    def __init__(self, model_size: str | None = None):
        # None (the common case -- no per-job override) falls back to the
        # global default, same behavior as before this class took a
        # constructor argument at all.
        self.model_size = model_size or settings.stt_local_model_size

    def transcribe(self, wav_path: str) -> SttResult:
        try:
            model = _get_model(self.model_size)
            segments_iter, info = model.transcribe(wav_path, vad_filter=True)
            segments = [
                SttSegment(start=seg.start, end=seg.end, text=seg.text.strip())
                for seg in segments_iter
            ]
        except Exception as exc:  # faster-whisper doesn't distinguish transient/permanent --
            # there's no network call to fail transiently, so any exception here
            # (corrupt audio, decode error, OOM) is treated as permanent.
            raise PermanentSttError(f"local whisper transcription failed: {exc}") from exc

        full_text = " ".join(s.text for s in segments if s.text).strip()
        return SttResult(segments=segments, full_text=full_text, language=getattr(info, "language", None))
