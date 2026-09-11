"""Deterministic candidate-clip scoring -- pure functions, no DB/IO, so this
is unit-testable without a real job/worker (see tests/test_scoring_logic.py).

Per the project's AI/ML principles: this is heuristic ranking, not a
virality model. Every feature here is a cheap, explainable signal computed
straight from the transcript + scene cuts already produced by earlier
stages -- no LLM call, no training data, no ground truth to fit against yet.
LLM/agent reasoning is allowed to *assist* later (better captions, hashtag
suggestions, an explanation of why a clip is strong) but must never be the
only thing choosing which clips survive; see app.workers.scoring for where
that assistive step would plug in (not implemented in the MVP).

Feature set follows the project's ranking guidance list. Two guidance items
are deliberately NOT implemented here and are called out below rather than
faked:
  - "visual motion proxy" -- would need real frame analysis (e.g. average
    optical-flow magnitude); out of scope for the MVP's CPU budget. Its
    weight defaults to 0.0 in app.core.config so it's inert until someone
    actually implements motion_proxy() and flips the weight.
  - "face/webcam presence" -- explicitly flagged in the project brief as
    "later if needed"; no feature function exists for it at all yet.

`hook_strength` is a proxy for watch-completion, not a new exception to the
"deterministic feature, no ML" rule above -- it's built the same way as
every other text/timing feature in this file. Rationale: TikTok's own
published ranking explanation (newsroom.tiktok.com/en-us/how-tiktok-
recommends-videos-for-you) names watch completion as "a strong indicator of
interest," and every credible third-party source on YouTube Shorts points
at the same lever -- neither platform documents follower count or past
video performance as a direct ranking input. A clip's completion rate is
driven heavily by whether the first couple of seconds hook the viewer or
lose them to dead air/a slow lead-in, so `hook_strength` scores exactly
that: how much of the window's own HOOK_WINDOW_SECONDS opening is actually
speech (not silence), plus a small bonus if that opening contains a
question or emotionally intense language (both classic "keep watching"
setups). This is still a heuristic proxy for one input to one platform's
partially-published ranking logic -- NOT a virality predictor, and it
cannot see or optimize for the platform's *other* signals (interactions,
follows, video metadata) at all. See app.core.config's
`score_weight_hook_strength` for the (currently unvalidated -- no real
approved/rejected clip outcomes to check it against yet) default weight.

`laughter`/`crowd_reaction` are the exception to "no ML here": they come
from an open-source pretrained audio classifier (PANNs -- see
app.core.audio_events), not text/scene-cut heuristics like everything else
in this file, so they can't be computed here (this module stays pure/no-IO
on purpose -- loading a ~300MB model and running audio inference is neither).
Same pattern as scene_changes/scene_cuts: the actual detection happens in
the impure worker (app.workers.scoring), which passes in an
already-computed `audio_events` dict per window, exactly like it already
does for `scene_cuts`. Still a fixed pretrained classifier's output, not a
generative/LLM signal -- same "deterministic feature, not an LLM choosing
clips" principle as everything else here. Opt-in
(ENABLE_AUDIO_EVENT_SCORING, default False); both raw values are 0.0 (fully
inert) when the caller doesn't pass an `audio_events` dict, so scoring
behaves identically to before this feature existed unless explicitly
turned on.

"emotional language" here means intensity/exclamation-style language (a
small hand-picked lexicon), NOT a profanity filter -- profanity detection
is a moderation/safety concern (see the project's "banned content flags"
requirement), not a scoring signal, and conflating the two would bias
ranking toward flagged content instead of just flagging it. If profanity
detection gets built, it belongs in a separate safety-check module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Words/phrases that tend to signal emotional intensity in spoken commentary
# (streamer/creator context) -- deliberately small and hand-picked, not a
# sentiment model. False negatives are fine (this is one signal of several);
# false positives on other lexicon entries matter more, so keep it short and
# unambiguous. Matched case-insensitively on word boundaries.
EMOTIONAL_LEXICON: frozenset[str] = frozenset(
    {
        "insane", "crazy", "unbelievable", "incredible", "amazing",
        "shocking", "terrifying", "hilarious", "furious", "devastated",
        "ecstatic", "wow", "omg", "no way", "what the", "can't believe",
        "never seen", "worst", "best ever", "so scared", "so happy",
        "clutch", "insane", "wtf", "goat", "unreal",
    }
)

# Raw-feature -> value that maps to a normalized score of 1.0. Chosen from
# rough intuition about a 15-90s clip window, not fitted to data -- there is
# no ground truth yet (see module docstring). Revisit once real jobs produce
# enough approved/rejected clips to sanity-check these against outcomes.
_NORMALIZATION_CAPS: dict[str, float] = {
    "speech_density": 3.0,       # words/sec -- fast, energetic talking
    "pause_count": 4.0,          # distinct speech/silence beats in the window
    "question_marks": 3.0,       # "?" occurrences -- setup/payoff structure
    "emotional_language": 4.0,   # lexicon hits
    "scene_changes": 3.0,        # visual cuts inside the window
    "motion_proxy": 1.0,         # not implemented; see module docstring
    "laughter": 1.0,             # already a 0..1 model probability -- no rescaling needed
    "crowd_reaction": 1.0,       # same
    "hook_strength": 1.0,        # already normalized 0..1 by hook_strength() -- no rescaling needed
}

_WORD_RE = re.compile(r"\w+")
# Gap between two consecutive transcript segments that counts as a distinct
# "beat" (silence-to-speech change) rather than just the normal micro-gap
# between words/sentences. Deliberately smaller than segmentation's
# SEGMENT_SILENCE_GAP_SECONDS (1.2s, a block-boundary signal) -- this is
# about counting rhythm *within* one candidate window, not splitting it.
_PAUSE_BEAT_SECONDS = 0.35

# How much of a candidate window's opening counts as "the hook" for
# hook_strength() below. 3s is short enough to actually be about the first
# beat (not just "the first third of a 90s clip"), long enough to catch a
# short opening sentence/question rather than penalizing normal speech
# cadence. A guess, like every other constant in this module -- no ground
# truth yet, see module docstring.
HOOK_WINDOW_SECONDS = 3.0
# Weight split between "opens on speech, not dead air" (immediacy) and "the
# opening is itself an attention-grabbing beat" (question/emotional
# language). Immediacy gets the larger share deliberately: dead air at the
# very start of a clip is the more unambiguous completion-rate killer of
# the two -- a clip that opens on ordinary but immediate speech should
# still score better than one that opens on several seconds of silence.
_HOOK_IMMEDIACY_WEIGHT = 0.7
_HOOK_ATTENTION_BONUS = 0.3


def _segments_in_window(segments: list[dict], start: float, end: float) -> list[dict]:
    """Transcript segments that overlap [start, end] at all, in order."""
    return [s for s in segments if s.get("end", 0) > start and s.get("start", 0) < end]


def window_text(segments: list[dict], start: float, end: float) -> str:
    """Concatenated transcript text for whatever falls inside the window."""
    return " ".join(s.get("text", "").strip() for s in _segments_in_window(segments, start, end)).strip()


def speech_density(segments: list[dict], start: float, end: float) -> float:
    """Words per second of window duration -- a fast talker/energetic
    moment scores higher than dead air or slow narration."""
    duration = end - start
    if duration <= 0:
        return 0.0
    text = window_text(segments, start, end)
    word_count = len(_WORD_RE.findall(text))
    return word_count / duration


def pause_count(segments: list[dict], start: float, end: float) -> int:
    """Number of >= _PAUSE_BEAT_SECONDS gaps between consecutive segments
    inside the window -- a proxy for comedic timing / dramatic beats
    ("silence-to-speech changes" in the project's ranking guidance)."""
    window_segments = sorted(_segments_in_window(segments, start, end), key=lambda s: s.get("start", 0))
    count = 0
    for prev, nxt in zip(window_segments, window_segments[1:]):
        gap = nxt.get("start", 0) - prev.get("end", 0)
        if gap >= _PAUSE_BEAT_SECONDS:
            count += 1
    return count


def question_mark_count(segments: list[dict], start: float, end: float) -> int:
    """Question marks in the window's text -- a rough proxy for
    question/payoff structure (project ranking guidance)."""
    return window_text(segments, start, end).count("?")


def emotional_language_hits(segments: list[dict], start: float, end: float) -> int:
    """Count of EMOTIONAL_LEXICON entries appearing in the window's text.
    Multi-word entries are matched as substrings (case-insensitive);
    single-word entries are matched on word boundaries so e.g. "wow" doesn't
    match inside "wowed" -- close enough for a heuristic, not NLP."""
    text = window_text(segments, start, end).lower()
    hits = 0
    for phrase in EMOTIONAL_LEXICON:
        if " " in phrase:
            hits += text.count(phrase)
        else:
            hits += len(re.findall(rf"\b{re.escape(phrase)}\b", text))
    return hits


def scene_change_count(scene_cuts: list[float], start: float, end: float) -> int:
    """Number of detected scene cuts landing inside the window -- visual
    variety, per the project's ranking guidance."""
    return sum(1 for cut in scene_cuts if start < cut < end)


def motion_proxy(segments: list[dict], start: float, end: float) -> float:  # noqa: ARG001
    """Not implemented -- see module docstring. Always 0.0; weight defaults
    to 0.0 in config so this is inert, not silently wrong."""
    return 0.0


def hook_strength(
    segments: list[dict], start: float, end: float, hook_window_seconds: float = HOOK_WINDOW_SECONDS
) -> float:
    """0..1 proxy for how strong this window's opening beat is -- see the
    module docstring's "hook_strength" section for the watch-completion
    rationale. Already normalized to 0..1 (like laughter/crowd_reaction),
    not a raw count -- _NORMALIZATION_CAPS caps it at 1.0 unchanged.

    Two components, blended:
      - immediacy: 1.0 if speech starts right at (or before) the window's
        own start, decaying linearly to 0.0 as dead air eats the whole hook
        window. A window with literally no speech in the hook range (a
        candidate that opens on total silence) scores 0.0 outright.
      - attention bonus: a flat bump if the opening text contains a
        question mark or a hit from EMOTIONAL_LANGUAGE -- both are classic
        "keep watching to find out" setups, not proof the hook is actually
        good (this can't read tone or delivery, only transcript text).
    """
    hook_end = min(start + hook_window_seconds, end)
    if hook_end <= start:
        return 0.0
    opening_segments = _segments_in_window(segments, start, hook_end)
    if not opening_segments:
        return 0.0  # dead air for the entire hook window -- worst case
    first_speech_start = min(s.get("start", start) for s in opening_segments)
    dead_air = max(0.0, first_speech_start - start)
    immediacy = max(0.0, 1.0 - dead_air / hook_window_seconds)
    opening_text = window_text(segments, start, hook_end)
    has_attention_grabber = "?" in opening_text or emotional_language_hits(segments, start, hook_end) > 0
    bonus = _HOOK_ATTENTION_BONUS if has_attention_grabber else 0.0
    return min(1.0, _HOOK_IMMEDIACY_WEIGHT * immediacy + bonus)


@dataclass(frozen=True)
class ScoreWeights:
    speech_density: float
    pause_count: float
    question_marks: float
    emotional_language: float
    scene_changes: float
    motion_proxy: float
    laughter: float = 0.0
    crowd_reaction: float = 0.0
    # Defaults to 0.0 here for the same reason laughter/crowd_reaction do --
    # so existing/test callers that construct ScoreWeights without naming
    # every field (e.g. ScoreWeights(0, 0, 0, 0, 0, 0) for "everything off")
    # keep getting exactly that. Unlike laughter/crowd_reaction, though,
    # hook_strength needs no new dependency and no extra impure input beyond
    # the transcript segments every caller already passes -- so it's turned
    # on by default at the config layer instead (see
    # app.core.config.score_weight_hook_strength, passed through
    # unconditionally -- not gated by any enable_* flag -- in
    # app.workers.scoring._weights_from_settings).
    hook_strength: float = 0.0


def score_window(
    segments: list[dict],
    start: float,
    end: float,
    scene_cuts: list[float],
    weights: ScoreWeights,
    audio_events: dict | None = None,
) -> tuple[float, dict]:
    """Compute the composite score for one candidate window plus a
    breakdown dict (raw feature values, normalized 0..1 values, and each
    feature's weighted contribution) -- the breakdown is stored verbatim on
    CandidateSegment.score_breakdown so a human reviewer (or a future "why
    was this picked" LLM explainer) can see exactly why a clip ranked where
    it did, instead of a single opaque number.

    `audio_events`: optional {"laughter": float, "crowd_reaction": float}
    (each a 0..1 model probability for this specific window), precomputed by
    app.workers.scoring via app.core.audio_events when
    ENABLE_AUDIO_EVENT_SCORING is on -- same "impure worker computes it once,
    passes the plain value into this pure function" pattern as `scene_cuts`.
    None (the default) means the feature wasn't computed for this window
    (flag off, or detection unavailable) -- both raw values are then 0.0,
    same as any other feature with weight 0.0: present in the breakdown,
    contributes nothing.
    """
    audio_events = audio_events or {}
    raw = {
        "speech_density": speech_density(segments, start, end),
        "pause_count": float(pause_count(segments, start, end)),
        "question_marks": float(question_mark_count(segments, start, end)),
        "emotional_language": float(emotional_language_hits(segments, start, end)),
        "scene_changes": float(scene_change_count(scene_cuts, start, end)),
        "motion_proxy": motion_proxy(segments, start, end),
        "laughter": float(audio_events.get("laughter", 0.0)),
        "crowd_reaction": float(audio_events.get("crowd_reaction", 0.0)),
        "hook_strength": hook_strength(segments, start, end),
    }

    return _composite_from_raw(raw, weights)


def _composite_from_raw(raw: dict[str, float], weights: ScoreWeights) -> tuple[float, dict]:
    """Normalize/weight/sum an already-computed raw feature dict into the
    (composite, breakdown) pair both scorers return.

    Shared by score_window and score_multipart_window so the two can never
    drift apart in how they weight or rescale -- the only thing that
    differs between a single-window and a stitched candidate is how the RAW
    feature values are measured (see score_multipart_window), and a
    reviewer comparing the two clips' 0..10 scores has to be comparing
    numbers produced the same way.
    """
    weight_map = {
        "speech_density": weights.speech_density,
        "pause_count": weights.pause_count,
        "question_marks": weights.question_marks,
        "emotional_language": weights.emotional_language,
        "scene_changes": weights.scene_changes,
        "motion_proxy": weights.motion_proxy,
        "laughter": weights.laughter,
        "crowd_reaction": weights.crowd_reaction,
        "hook_strength": weights.hook_strength,
    }

    normalized: dict[str, float] = {}
    contributions: dict[str, float] = {}
    composite = 0.0
    for feature, raw_value in raw.items():
        cap = _NORMALIZATION_CAPS[feature]
        norm_value = min(raw_value / cap, 1.0) if cap > 0 else 0.0
        contribution = norm_value * weight_map[feature]
        normalized[feature] = norm_value
        contributions[feature] = contribution
        composite += contribution

    breakdown = {
        "raw": raw,
        "normalized": normalized,
        "weights": weight_map,
        "contributions": contributions,
        "composite": composite,
    }
    breakdown["score_out_of_10"] = score_out_of_10(breakdown)
    return composite, breakdown


def score_multipart_window(
    segments: list[dict],
    parts: list[tuple[float, float]],
    scene_cuts: list[float],
    weights: ScoreWeights,
    audio_events: dict | None = None,
) -> tuple[float, dict]:
    """score_window's equivalent for a stitched multi-part candidate (see
    app.core.rendering_logic.normalize_parts) -- features computed over the
    UNION of the parts, not the span they cover.

    Scoring a stitched clip by its outer [first start, last end] span would
    be actively wrong: the whole point of a multi-part clip is that the
    material between the parts was cut out, so a span-based speech_density
    would divide real words by a duration that includes minutes the viewer
    never sees, and pause_count/scene_changes would count beats that happen
    entirely inside the removed gaps.

    Per-feature semantics, each chosen to match what the viewer actually
    experiences:
      - speech_density: total words across parts over total PLAYING time.
      - pause_count / question_marks / emotional_language / scene_changes:
        summed within each part. Deliberately NOT counted across a join --
        the cut between two parts isn't a dramatic pause the creator timed,
        it's an edit, and crediting it would reward stitching for its own
        sake.
      - hook_strength: computed on the FIRST part only. It is a proxy for
        "do the opening seconds hold a viewer" (see hook_strength), and the
        opening of a stitched clip is the opening of its first part.
      - motion_proxy/laughter/crowd_reaction: unchanged -- the first is
        still unimplemented, and the audio-event values are already
        computed per candidate by the worker rather than derived here.
    """
    audio_events = audio_events or {}
    total_duration = sum(end - start for start, end in parts)
    total_words = sum(len(_WORD_RE.findall(window_text(segments, s, e))) for s, e in parts)

    raw = {
        "speech_density": (total_words / total_duration) if total_duration > 0 else 0.0,
        "pause_count": float(sum(pause_count(segments, s, e) for s, e in parts)),
        "question_marks": float(sum(question_mark_count(segments, s, e) for s, e in parts)),
        "emotional_language": float(sum(emotional_language_hits(segments, s, e) for s, e in parts)),
        "scene_changes": float(sum(scene_change_count(scene_cuts, s, e) for s, e in parts)),
        "motion_proxy": 0.0,
        "laughter": float(audio_events.get("laughter", 0.0)),
        "crowd_reaction": float(audio_events.get("crowd_reaction", 0.0)),
        "hook_strength": hook_strength(segments, parts[0][0], parts[0][1]),
    }
    return _composite_from_raw(raw, weights)


def score_out_of_10(breakdown: dict) -> float:
    """The composite score, rescaled to a fixed 0..10 range for display.

    `composite` alone isn't meaningfully bounded -- its ceiling is
    sum(weights) (only reached if every feature hits its normalization cap
    simultaneously), which moves whenever SCORE_WEIGHT_* settings change.
    Dividing by that same ceiling before scaling to 10 keeps the number a
    reviewer sees stable and comparable across jobs/weight configs, instead
    of being an opaque number whose "good" range depends on config a
    reviewer never sees. This is still the same deterministic heuristic --
    rescaling, not a different scoring model (see module docstring: this is
    ranking, not a virality prediction).
    """
    weights_sum = sum(breakdown["weights"].values())
    if weights_sum <= 0:
        return 0.0
    return round(min(max(breakdown["composite"] / weights_sum * 10.0, 0.0), 10.0), 2)


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection-over-union of two time windows -- 0 (disjoint) to 1
    (identical). Standard overlap metric for non-max suppression."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0


def select_top_non_overlapping(
    candidates: list[dict],
    *,
    max_count: int,
    min_score: float = 0.0,
    overlap_threshold: float = 0.5,
) -> set:
    """Greedy non-max suppression: picks up to `max_count` candidates by
    score, skipping any candidate that overlaps (IoU > overlap_threshold) an
    already-selected higher-scoring one.

    Needed once candidate generation can produce overlapping windows (see
    app.core.segmentation_logic's sliding-window sub-windows for long speech
    blocks) -- without this, a naive "take the top N by score" could return
    several near-duplicate clips of the same moment (e.g. windows 5s apart
    covering the same punchline) instead of N *distinct* highlights, which
    is a worse result for a reviewer than fewer, more varied clips.

    `candidates`: list of {"id": ..., "start": float, "end": float,
    "score": float}. Returns the set of selected ids (order not
    significant -- callers already have the full candidate list to look up
    from).
    """
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    selected: list[dict] = []
    for candidate in ordered:
        if candidate["score"] < min_score:
            continue
        if len(selected) >= max_count:
            break
        if any(
            _iou(candidate["start"], candidate["end"], s["start"], s["end"]) > overlap_threshold
            for s in selected
        ):
            continue
        selected.append(candidate)
    return {c["id"] for c in selected}


def default_caption(segments: list[dict], start: float, end: float, max_chars: int = 140) -> str:
    """A first-pass caption: the window's own transcript text, truncated to
    a whole word under max_chars. This is a placeholder a human reviewer can
    edit before upload, not a claim that it's a good caption -- an
    LLM-assisted caption/hashtag generator is explicitly a post-MVP
    enhancement per the project's AI/ML principles (LLM may assist with
    "writing captions", but the deterministic default must work without it).
    """
    text = window_text(segments, start, end)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{truncated}..." if truncated else text[:max_chars]
