"""LLM-assisted clip title/hashtag/caption generation -- the
network-calling half of app.core.caption_logic (see that module's
docstring for the split rationale, the full field list, and why title
generation now has to happen before rendering).

Two providers, both spoken through the `openai` package's chat-completions
client (CAPTION_LLM_PROVIDER):
  - "openai": the real OpenAI API, reuses OPENAI_API_KEY (already required
    for STT_PROVIDER=openai too).
  - "ollama": a self-hosted open-weight model via Ollama's OpenAI-compatible
    endpoint (http://localhost:11434/v1 by default) -- same client, same
    prompt, same response parsing, just a different base_url and no real
    API key (Ollama doesn't check it; the openai client just requires the
    field be non-empty). No new provider abstraction/interface needed since
    both cases are "call the OpenAI-shaped chat completions API," just
    against a different endpoint -- see app.core.stt's actual
    provider-interface split for a case where the two implementations
    genuinely don't share a call shape and that pattern earns its keep.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.core.caption_logic import (
    build_caption_prompt,
    heuristic_explanation,
    heuristic_hashtags,
    heuristic_title,
    parse_caption_response,
)
from app.core.config import settings
from app.core.scoring_logic import window_text
from app.workers.common import logger


def _heuristic_annotation(score_breakdown: dict | None, existing_caption: str, reason: str) -> dict:
    return {
        "source": "heuristic_fallback",
        "reason": reason,
        "title": heuristic_title(score_breakdown),
        "hashtags": heuristic_hashtags(score_breakdown),
        "caption": existing_caption,
        "explanation": heuristic_explanation(score_breakdown),
        "model": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_caption_annotation(
    transcript_segments: list[dict],
    start: float,
    end: float,
    score_breakdown: dict | None,
    existing_caption: str,
    llm_config=None,
) -> dict:
    """Best-effort LLM title/hashtags/caption/explanation for one clip.
    Always returns a usable annotation dict -- never raises -- degrading to
    _heuristic_annotation (built entirely from data already on hand, no
    network call) when LLM captions are disabled, no API key is
    configured, or the call/response itself fails for any reason. Called
    synchronously from app.workers.rendering, before the ffmpeg filtergraph
    is built, so the returned "title" can be burned into the clip itself
    (app.core.rendering_logic.build_title_srt) -- unlike hashtags/caption/
    explanation, which are metadata only, the title can't be generated
    after the fact without a full re-render. app.workers.caption_generation
    still exists as a manual regenerate path (e.g. to redo an annotation
    without re-rendering the video), just no longer auto-enqueued.
    """
    if not settings.enable_llm_captions:
        return _heuristic_annotation(score_breakdown, existing_caption, "disabled")

    if llm_config is None:
        from app.core.llm_config import resolve_llm_config

        llm_config = resolve_llm_config(None)
    if not llm_config.usable:
        return _heuristic_annotation(score_breakdown, existing_caption, "no_api_key")
    provider = llm_config.provider

    if provider == "ollama":
        base_url = llm_config.base_url
        api_key = llm_config.api_key  # Ollama ignores it; the openai client needs a non-empty string
        model = llm_config.model
    else:
        base_url = None  # openai package's own default (OpenAI's real API)
        api_key = llm_config.api_key
        model = llm_config.model

    clip_text = window_text(transcript_segments, start, end)
    hint = heuristic_explanation(score_breakdown)
    prompt = build_caption_prompt(clip_text, hint, existing_caption)

    try:
        import openai  # local import: keep the dependency optional when LLM captions are disabled

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        raw_text = response.choices[0].message.content
        parsed = parse_caption_response(raw_text)
    except Exception as exc:
        # Covers a down/unreachable Ollama server the same way it already
        # covered OpenAI errors -- connection failures land here too, not
        # just malformed responses, so switching providers never trades
        # "no title" for a hard pipeline failure.
        logger.warning("caption_generation.llm_failed", provider=provider, error=str(exc))
        return _heuristic_annotation(score_breakdown, existing_caption, f"llm_error: {exc}"[:300])

    return {
        "source": "llm",
        "reason": None,
        "title": parsed["title"],
        "hashtags": parsed["hashtags"],
        "caption": parsed["caption"],
        "explanation": parsed["explanation"],
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
