"""Pure helpers for LLM-assisted clip hashtag/caption generation -- prompt
building and response parsing/validation, same split as the rest of
app/core/*_logic.py (keep the parts that are just string/dict manipulation
out of the network-calling module so they're fast to unit test without
mocking an API). See app.core.caption_generation for the actual OpenAI
call, and app.workers.caption_generation for the worker that wires this
into the pipeline.

This generates the *social post* hashtags/caption/explanation AND, as of
the clickbait-title feature, the burned-in on-screen **title** banner's
text too (app.core.rendering_logic.build_title_srt turns that text into
the actual subtitle cue app.workers.rendering burns in -- this module only
produces/validates the text itself, same split as the transcript captions'
build_srt). All four fields are stored together in
CandidateSegment.llm_annotation. Matches the project's AI/ML principle that
an LLM may assist with "writing captions, generating hashtags, summarizing
clip context" but must never choose which clips get made -- see
app.core.scoring_logic for the (unrelated, deterministic) ranking step this
plugs in after, not instead of.

The title needs to exist *before* rendering (it gets burned into the
clip's pixels, unlike hashtags/caption/explanation which are metadata
only) -- see app.workers.rendering.run, which now calls
app.core.caption_generation.generate_caption_annotation synchronously
before building the ffmpeg filtergraph, instead of the old
after-render-async design (app.workers.caption_generation is kept around
as a manual regenerate path, not auto-invoked by rendering anymore).
"""
from __future__ import annotations

import json

MAX_CAPTION_CHARS = 150
MAX_EXPLANATION_CHARS = 300
MAX_TITLE_CHARS = 70
MAX_HASHTAGS = 6
MIN_HASHTAGS = 3

# Human-readable phrasing for scoring's raw feature keys -- used to build a
# deterministic explanation (and hint the LLM prompt) from the same
# score_breakdown the reviewer already sees (app.core.scoring_logic
# .score_window). Kept in sync by hand since the feature set changes rarely.
_FEATURE_PHRASES: dict[str, str] = {
    "speech_density": "fast-paced talking",
    "pause_count": "a distinct comedic/dramatic beat",
    "question_marks": "a question-and-payoff moment",
    "emotional_language": "high emotional/reactive language",
    "scene_changes": "a visual scene change",
    "motion_proxy": "high on-screen motion",
}


def heuristic_explanation(score_breakdown: dict | None, top_n: int = 2) -> str:
    """Deterministic 'what happens / why recommended' sentence built from
    the same per-feature contributions score_window already computed --
    the non-LLM fallback for CandidateSegment.llm_annotation['explanation'],
    and also used as a grounding hint in the LLM prompt itself. Never
    returns an empty string.
    """
    contributions = (score_breakdown or {}).get("contributions") or {}
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    top = [key for key, value in ranked[:top_n] if value > 0]
    if not top:
        return "Selected by the ranking algorithm; no single signal stood out strongly."
    phrases = [_FEATURE_PHRASES.get(key, key) for key in top]
    return "Selected mainly for " + " and ".join(phrases) + "."


# Punchy top-of-frame "hook" title templates, keyed by the same
# score_breakdown contribution keys as _FEATURE_PHRASES above -- a
# different vocabulary on purpose, since this one has to work as a
# clickbait-style hook a viewer reads in under a second at the top of the
# frame, not a reviewer-facing explanation sentence.
_TITLE_TEMPLATES: dict[str, str] = {
    "speech_density": "He Would NOT Stop Talking About This",
    "pause_count": "Then... Dead Silence",
    "question_marks": "Wait, Did That Actually Just Happen?",
    "emotional_language": "You Won't Believe His Reaction",
    "scene_changes": "Everything Changed In One Second",
    "motion_proxy": "The Chaos Was Unreal",
}
_DEFAULT_TITLE = "You Have To See This Clip"


def heuristic_title(score_breakdown: dict | None) -> str:
    """Deterministic clickbait-style hook title -- the non-LLM fallback for
    the burned-in on-screen title banner (top of frame; see
    app.core.rendering_logic.build_title_srt and
    app.workers.rendering._TITLE_STYLE). Keyed off score_breakdown's single
    top-contributing signal (not top_n like heuristic_explanation -- a
    title has to commit to one hook, not a list). Never returns an empty
    string, so a clip is never left without a burned-in title just because
    the LLM path is unavailable.
    """
    contributions = (score_breakdown or {}).get("contributions") or {}
    top = max(contributions, key=contributions.get, default=None) if contributions else None
    return _TITLE_TEMPLATES.get(top, _DEFAULT_TITLE)


def heuristic_hashtags(score_breakdown: dict | None) -> list[str]:
    """Deterministic hashtag fallback -- generic but always present, so a
    clip is never left with zero hashtags just because the LLM path is
    unavailable. Adds a signal-specific tag or two on top of the always-on
    base set when a raw feature clearly stood out, using the same
    contributions data as heuristic_explanation.
    """
    tags = ["#clip", "#streamer", "#twitchclips"]
    contributions = (score_breakdown or {}).get("contributions") or {}
    top = max(contributions, key=contributions.get, default=None) if contributions else None
    extra = {
        "emotional_language": "#omg",
        "question_marks": "#wait",
        "scene_changes": "#insane",
        "speech_density": "#viral",
    }.get(top)
    if extra:
        tags.append(extra)
    return tags


def normalize_hashtags(raw_tags: list, max_hashtags: int = MAX_HASHTAGS) -> list[str]:
    """Shared normalization for a list of raw hashtag strings, from either
    the LLM response (parse_caption_response below) or a reviewer's manual
    edit (app.api.routers.clips' PATCH .../caption) -- strips whitespace and
    a leading '#' (re-added consistently), drops blanks/duplicates
    (case-sensitive; "#OMG" and "#omg" are kept distinct on purpose, since
    hashtag capitalization is sometimes meaningful), and caps the count.
    Never raises -- an all-blank/empty input just returns [].
    """
    tags: list[str] = []
    for tag in raw_tags:
        tag = str(tag).strip().lstrip("#").replace(" ", "")
        if tag and f"#{tag}" not in tags:
            tags.append(f"#{tag}")
        if len(tags) >= max_hashtags:
            break
    return tags


def build_caption_prompt(clip_text: str, explanation_hint: str, existing_caption: str) -> str:
    """The user-turn prompt for the hashtag/caption/title-generation LLM
    call. Grounded entirely in this clip's own transcript text
    (`clip_text`) -- hashtags/captions/titles are generated from what the
    creator actually said/did in this specific clip, not a learned
    personal voice profile (no such profile exists in this schema; a
    future "creator style" feature would be its own addition, not folded
    in here).
    """
    clip_text = (clip_text or "").strip() or "(no speech detected in this window)"
    return (
        "You are helping a streamer/creator prepare one short-form vertical "
        "video clip for review before posting to TikTok. Below is the clip's "
        "own transcript and a note on why the ranking algorithm picked it.\n\n"
        f'Transcript:\n"""\n{clip_text}\n"""\n\n'
        f"Ranking note: {explanation_hint}\n\n"
        f'Fallback caption (use only if you cannot do better): "{existing_caption}"\n\n'
        "Reply with ONLY a JSON object, no markdown fences, with exactly these keys:\n"
        '  "title": a short, punchy, clickbait-style hook (<70 chars) for a bold banner burned into the TOP of the video for its whole duration -- this is the first thing a viewer reads, so make it grab attention (e.g. "He Did NOT See That Coming"), but it must stay honest to what actually happens in the transcript, never a hook for something that never happens\n'
        '  "hashtags": a JSON array of 3-6 relevant hashtags as plain words or short phrases WITHOUT the leading \'#\' (it will be added) -- be specific to what actually happens in the transcript, not just generic streaming tags\n'
        '  "caption": a short punchy TikTok-style caption/hook (<150 chars) for the post itself (separate from the on-screen title above), grounded in what actually happens in the transcript, not generic hype\n'
        '  "explanation": one or two sentences for a human reviewer on what happens in the clip and why it is recommended for posting\n'
    )


def parse_caption_response(raw_text: str) -> dict:
    """Validate/clamp the LLM's JSON reply into the stored annotation
    shape. Raises ValueError on anything malformed -- callers (app.core
    .caption_generation) catch that and fall back to the heuristic
    annotation rather than trusting a response that doesn't match the
    contract asked for.
    """
    raw_text = (raw_text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"caption response was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("caption response JSON was not an object")

    hashtags_raw = data.get("hashtags")
    if not isinstance(hashtags_raw, list) or not hashtags_raw:
        raise ValueError("caption response 'hashtags' was not a non-empty list")

    hashtags = normalize_hashtags(hashtags_raw)
    if not hashtags:
        raise ValueError("caption response 'hashtags' had no usable entries")

    title = str(data.get("title") or "").strip()
    caption = str(data.get("caption") or "").strip()
    explanation = str(data.get("explanation") or "").strip()
    if not title:
        raise ValueError("caption response missing non-empty 'title'")
    if not caption:
        raise ValueError("caption response missing non-empty 'caption'")

    return {
        "title": title[:MAX_TITLE_CHARS],
        "hashtags": hashtags,
        "caption": caption[:MAX_CAPTION_CHARS],
        "explanation": explanation[:MAX_EXPLANATION_CHARS],
    }
