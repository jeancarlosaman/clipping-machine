"""LLM-assisted candidate-window *suggestion* -- the network-calling half
of app.core.llm_segmentation_logic (see that module's docstring for the
prompt/parsing rationale). Reuses the exact same OpenAI-shaped
chat-completions client and provider settings as app.core
.caption_generation (CAPTION_LLM_PROVIDER/CAPTION_LLM_MODEL/
CAPTION_OLLAMA_MODEL/CAPTION_OLLAMA_BASE_URL) -- see
ENABLE_LLM_SEGMENT_SUGGESTIONS's comment in app/core/config.py for why this
doesn't get its own parallel provider config.

Architecturally this is an ADDITIVE candidate proposer, not a decider: it
reads a job's transcript once and returns a list of suggested windows with
reasons; app/workers/segmentation.py inserts those as extra
`candidate_segments` rows (origin="llm") alongside whatever the
deterministic sliding-window segmentation already found (origin=
"heuristic") -- both land in the same pending_score pool, and the existing
deterministic scoring + non-max-suppression selection in
app/workers/scoring.py picks winners from the combined set exactly like it
already does today. This module never chooses which clips get made; it
only ever adds more candidates for that existing, unmodified decision
process to consider.

Same "never raises" posture as every other optional signal in this
codebase (app.core.caption_generation's heuristic fallback,
app.core.audio_events' zero-on-failure) -- disabled, no API key,
unreachable provider, a malformed response, or a pathologically long
transcript (see ENABLE_LLM_SEGMENT_SUGGESTIONS' sibling settings) all
degrade to an empty list of suggestions, never a failed segmentation job.

Post-MVP idea, NOT implemented here: a video-native model (e.g. Gemini's
Files API) that watches the actual video instead of just reading the
transcript, so it could also catch silent/visual-only highlights (a
reaction with no speech, a pure-gameplay moment) that a transcript-only
pass is blind to. That's a real, heavier upgrade path -- new SDK, new API
key, materially higher cost/latency for a full VOD, and its own
timestamp-snapping problem to solve (video-LLM timestamps drift too, so
it would need the same "snap to a real boundary, never trust the raw
number" treatment this module already applies to transcript indices) --
worth revisiting once this transcript-only version has proven the pattern
is useful at all.
"""
from __future__ import annotations

from app.core.config import settings
from app.core.llm_segmentation_logic import build_segment_suggestion_prompt, parse_segment_suggestions
from app.workers.common import logger


def generate_llm_segment_suggestions(
    transcript_segments: list[dict],
    min_len: float,
    max_len: float,
    feedback_examples: list[dict] | None = None,
) -> list[dict]:
    """Best-effort list of LLM-suggested candidate windows for one job's
    transcript, each `{"start": float, "end": float, "reason": str}`.
    Always returns a list -- never raises -- degrading to [] when the
    feature is disabled, no provider is available, the transcript is too
    long to safely prompt with (see llm_segment_max_transcript_segments),
    or the call/response itself fails for any reason. Called once per
    segmentation job, from app/workers/segmentation.run, after the
    deterministic sliding-window pass has already produced its own
    candidates -- this never runs instead of that pass, only alongside it.

    `feedback_examples`: optional worked examples of this creator's own
    past approve/reject decisions and the comments they wrote on them,
    gathered by the caller (the worker owns the DB query; this module
    stays a pure "prompt + call + parse" step). Passed straight through to
    build_segment_suggestion_prompt, which renders them as a few-shot
    block. Empty/None prompts exactly as before the feedback loop existed.
    """
    if not settings.enable_llm_segment_suggestions:
        return []
    if not transcript_segments:
        return []
    if len(transcript_segments) > settings.llm_segment_max_transcript_segments:
        logger.warning(
            "llm_segmentation.transcript_too_long",
            segment_count=len(transcript_segments),
            limit=settings.llm_segment_max_transcript_segments,
        )
        return []

    provider = settings.caption_llm_provider
    if provider == "openai" and not settings.openai_api_key:
        return []

    if provider == "ollama":
        base_url = settings.caption_ollama_base_url
        api_key = "ollama"  # Ollama ignores this; the openai client just requires a non-empty string
        model = settings.caption_ollama_model
    else:
        base_url = None  # openai package's own default (OpenAI's real API)
        api_key = settings.openai_api_key
        model = settings.caption_llm_model

    max_suggestions = settings.llm_segment_max_suggestions
    prompt = build_segment_suggestion_prompt(
        transcript_segments, min_len, max_len, max_suggestions, feedback_examples=feedback_examples
    )

    try:
        import openai  # local import: keep the dependency optional when this feature is disabled

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # lower than captions' 0.7 -- this is a selection task, not creative writing
            max_tokens=1200,  # a full array of up to max_suggestions reasoned entries needs more room than one caption
        )
        raw_text = response.choices[0].message.content
        suggestions = parse_segment_suggestions(raw_text, transcript_segments, min_len, max_len, max_suggestions)
    except Exception as exc:
        # Covers a down/unreachable Ollama server the same way it already
        # covers OpenAI errors -- connection failures land here too, not
        # just malformed responses.
        logger.warning("llm_segmentation.llm_failed", provider=provider, error=str(exc))
        return []

    logger.info("llm_segmentation.suggested", provider=provider, model=model, count=len(suggestions))
    return suggestions
