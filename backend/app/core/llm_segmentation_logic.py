"""Pure helpers for the LLM segment-suggestion feature -- prompt building
and response parsing/validation, same split as app.core.caption_logic vs.
app.core.caption_generation (keep the parts that are just string/dict
manipulation out of the network-calling module so they're fast to unit
test without mocking an API). See app.core.llm_segmentation for the actual
LLM call and app.workers.segmentation for how these suggestions get merged
into the deterministic candidate pool.

Timestamp-hallucination mitigation (the whole reason this module exists
separately from a simpler "ask the LLM for start/end seconds" design): an
LLM asked to emit raw timestamps for a long transcript will confidently
invent numbers that don't line up with any real transcript segment,
especially past the first few minutes. Instead, the prompt shows each
transcript segment with its own INDEX and asks the LLM to reference
indices only; the real start/end seconds then come from
transcript_segments[index] itself, never from a number the LLM typed. A
response can still name an out-of-range or nonsensical index -- that's
handled by dropping just that one suggestion, never the whole batch (see
parse_segment_suggestions).
"""
from __future__ import annotations

import json

MAX_REASON_CHARS = 200
MAX_TOPIC_CHARS = 160
_DEFAULT_REASON = "Suggested by LLM analysis of the transcript."


def build_segment_suggestion_prompt(
    transcript_segments: list[dict],
    min_len: float,
    max_len: float,
    max_suggestions: int,
    feedback_examples: list[dict] | None = None,
) -> str:
    """User-turn prompt for the segment-suggestion LLM call. Grounded
    entirely in this job's own transcript -- each line is numbered with
    the index the LLM must reference back (see module docstring).

    Deliberately asks for SUBSTANCE, not just energy. The original version
    of this prompt listed only affect-based cues ("big reactions, jokes,
    dramatic reveals, hot takes, arguments, surprising or high-energy
    moments") and never once asked what was actually *said* -- which
    reliably surfaced the loudest moments rather than the most interesting
    ones. The categories below put content-bearing moments (a concrete
    claim, an opinion with its reasoning, a story with a payoff, a real
    explanation) on equal footing with reactions, and the required `topic`
    field forces the model to state the actual point in its own words
    before it can justify the pick -- a moment it can't summarize
    concretely is usually one that only sounded interesting.

    Two more short-form-specific requirements the old prompt never stated:
    the clip has to make sense to someone who hasn't seen the rest of the
    VOD, and it has to start at the setup rather than the punchline
    (opening mid-thought is the classic failure mode of automatic
    clipping).

    `feedback_examples`: optional list of this creator's own past review
    decisions -- {"decision": "approved"|"rejected", "notes": str,
    "transcript": str} -- rendered as worked examples of what they
    actually keep and cut. This is the project's "learn from my feedback"
    path without a training run: few-shot examples work at ten examples,
    where fine-tuning a selection/ranking model would want hundreds plus
    real negatives (see README's "Training on your own approved clips").
    """
    lines = []
    for i, seg in enumerate(transcript_segments):
        text = (seg.get("text") or "").strip()
        lines.append(f"[{i}] {float(seg['start']):.2f}-{float(seg['end']):.2f}: {text}")
    transcript_block = "\n".join(lines)

    feedback_block = build_feedback_examples_block(feedback_examples)

    return (
        "You are helping find compelling short-form clip moments in a streamer/creator's VOD "
        "transcript. Below is the full transcript, broken into numbered segments.\n\n"
        f"Transcript segments:\n{transcript_block}\n\n"
        f"{feedback_block}"
        f"Suggest up to {max_suggestions} distinct moments that would make strong standalone "
        "short-form clips. Judge them on WHAT IS ACTUALLY SAID, not just how loud or energetic "
        "the delivery is. Strong candidates include:\n"
        "  - a specific claim, fact, number or detail that is surprising or worth knowing\n"
        "  - a strong opinion together with the reasoning behind it (not just the hot take alone)\n"
        "  - a story or anecdote that has a setup and an actual payoff\n"
        "  - a clear explanation of something non-obvious\n"
        "  - a genuine disagreement, correction, or change of mind\n"
        "  - a joke, big reaction or dramatic reveal -- but only when the content behind it lands, "
        "not merely because the reaction is loud\n\n"
        "Requirements for every moment you pick:\n"
        "  - It must be self-contained: understandable to someone who has not seen the rest of "
        "this VOD, with no missing context.\n"
        "  - It must start at the SETUP, not the punchline -- include the lead-in that makes the "
        "moment land. Never start mid-sentence or mid-thought.\n"
        "  - Skip filler, small talk, stream logistics, greetings and reading chat aloud, however "
        "energetic they sound.\n\n"
        "For each moment, pick a start_index and end_index from the numbered list above marking "
        "exactly where the clip should start and end (inclusive) -- use the bracketed index "
        "numbers shown, NEVER timestamps or seconds. Prefer moments whose span works out to "
        f"roughly {min_len:.0f}-{max_len:.0f} seconds. Do not suggest overlapping duplicates of "
        "the same moment.\n\n"
        "Occasionally the material for one clip is split across parts of the VOD that are not "
        "next to each other -- a story told in two passes, or a payoff that calls back to "
        "something said much earlier. When (and ONLY when) joining separated stretches makes a "
        "genuinely better single clip than either stretch alone, you may return \"parts\": a list "
        "of 2 or 3 non-overlapping index ranges, in the order they should play. Their combined "
        f"length must still land in roughly {min_len:.0f}-{max_len:.0f} seconds. Each part must "
        "stand on its own as a complete thought -- the viewer sees a hard cut between them, so a "
        "part that ends mid-sentence will sound broken. Most moments should NOT use this; prefer "
        "a single continuous range whenever one works.\n\n"
        "Reply with ONLY a JSON array (no markdown fences, no other text). Each element must be an "
        "object with these keys:\n"
        '  "start_index": <int, index from the list above>\n'
        '  "end_index": <int, index from the list above, >= start_index>\n'
        '  "topic": what is actually said in this moment, stated concretely in one short phrase '
        '(e.g. "explains why he refunded the sponsor" -- NOT "a funny moment")\n'
        '  "reason": a one-sentence explanation of why this would work as a standalone clip\n'
        '  "parts": OPTIONAL, only for a stitched moment -- a list of 2-3 objects, each with its '
        'own "start_index" and "end_index", in playback order. Omit this key entirely for a '
        "normal single-range moment. When present, still fill in start_index/end_index as the "
        "first part's start and the last part's end.\n\n"
        "If nothing in this transcript genuinely stands out, reply with an empty array: []"
    )


def build_feedback_examples_block(feedback_examples: list[dict] | None) -> str:
    """Render this creator's own past approve/reject decisions (with their
    written comments) as a few-shot block for the prompt above, or "" when
    there are none -- so a fresh install prompts exactly as it did before
    this feature existed.

    Kept as its own function (rather than inlined) because it's the piece
    most worth unit-testing on its own: it's the only part of the prompt
    built from user-supplied free text, and an empty/whitespace comment or
    a missing transcript excerpt must degrade to "skip this example"
    rather than emitting a confusing half-example.
    """
    if not feedback_examples:
        return ""

    rendered: list[str] = []
    for example in feedback_examples:
        notes = str(example.get("notes") or "").strip()
        if not notes:
            # No comment means no signal about *why* -- an approve/reject
            # with no reasoning teaches the model nothing here (the
            # deterministic scorer already handles "was it picked").
            continue
        decision = str(example.get("decision") or "").strip().lower()
        verdict = "KEPT" if decision == "approved" else "CUT"
        transcript = " ".join(str(example.get("transcript") or "").split())[:400]
        if transcript:
            rendered.append(f'- {verdict} -- clip said: "{transcript}"\n  their comment: "{notes}"')
        else:
            rendered.append(f'- {verdict} -- their comment: "{notes}"')

    if not rendered:
        return ""

    return (
        "This creator has reviewed clips from earlier VODs and left comments explaining their "
        "decisions. Use these to calibrate what they personally consider clip-worthy -- their "
        "taste overrides the general guidance below wherever the two disagree:\n"
        + "\n".join(rendered)
        + "\n\n"
    )


def parse_segment_suggestions(
    raw_text: str,
    transcript_segments: list[dict],
    min_len: float,
    max_len: float,
    max_suggestions: int,
) -> list[dict]:
    """Validate/clamp the LLM's JSON reply into a list of
    {"start": float, "end": float, "reason": str} dicts, using ONLY the
    real start/end seconds already on transcript_segments -- never a
    number the LLM emitted directly (see module docstring).

    Raises ValueError if the response isn't a JSON array at all (the
    caller, app.core.llm_segmentation, catches that and degrades to zero
    suggestions same as any other failure). An individual malformed
    element (bad indices, non-dict, out-of-range duration, etc.) is just
    skipped -- one bad suggestion should never discard the rest of an
    otherwise-useful response.
    """
    raw_text = (raw_text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"segment suggestion response was not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("segment suggestion response JSON was not an array")

    n = len(transcript_segments)
    results: list[dict] = []
    seen: set[tuple[float, float]] = set()

    for item in data:
        if len(results) >= max_suggestions:
            break
        if not isinstance(item, dict):
            continue

        # A stitched multi-part moment, when the model returned one and it
        # validates. Falls through to the ordinary single-range path on
        # anything malformed -- start_index/end_index are required either
        # way (the prompt asks for them as the overall span even when
        # parts is present), so a bad `parts` costs the suggestion its
        # stitching, never the suggestion itself.
        parts = _parse_parts(item.get("parts"), transcript_segments, min_len, max_len)
        if parts is not None:
            start, end = parts[0][0], parts[-1][1]
            key = (round(start, 1), round(end, 1))
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "start": start,
                    "end": end,
                    "reason": _combined_reason(item),
                    "parts": [[p_start, p_end] for p_start, p_end in parts],
                }
            )
            continue

        try:
            start_index = int(item.get("start_index"))
            end_index = int(item.get("end_index"))
        except (TypeError, ValueError):
            continue
        if not (0 <= start_index < n and 0 <= end_index < n and start_index <= end_index):
            continue

        start = float(transcript_segments[start_index]["start"])
        end = float(transcript_segments[end_index]["end"])
        duration = end - start
        if duration < min_len or duration > max_len:
            # Silently dropped, not clamped -- stretching/shrinking a
            # window the LLM picked for a specific reason could make it no
            # longer match that reason. Scoring only ever sees windows
            # that already satisfy this job's own min/max, same as every
            # heuristic candidate.
            continue

        key = (round(start, 1), round(end, 1))
        if key in seen:
            continue
        seen.add(key)

        results.append({"start": start, "end": end, "reason": _combined_reason(item)})

    return results


MAX_PARTS = 3


def _parse_parts(
    raw_parts, transcript_segments: list[dict], min_len: float, max_len: float
) -> list[tuple[float, float]] | None:
    """Validate a response's optional `parts` list into real (start, end)
    second pairs, or None if it isn't a usable stitched moment.

    Same index-only discipline as the single-range path (see module
    docstring): each part names transcript indices and the real seconds
    come from transcript_segments, never from a number the model typed.

    Rejects, in every case by returning None so the caller falls back to
    treating the suggestion as an ordinary single range:
      - fewer than 2 or more than MAX_PARTS parts (one part isn't a stitch;
        many parts is a montage, which is a different feature and much
        easier to get incoherently wrong)
      - any out-of-range/backwards index
      - parts that overlap or are given out of order -- the prompt asks for
        playback order, and silently re-sorting could invert a
        setup->payoff the model deliberately sequenced
      - a COMBINED playing time outside the job's [min_len, max_len], which
        is the sum of the parts, not the span they cover
    """
    if not isinstance(raw_parts, (list, tuple)):
        return None
    if not (2 <= len(raw_parts) <= MAX_PARTS):
        return None

    n = len(transcript_segments)
    parts: list[tuple[float, float]] = []
    for entry in raw_parts:
        if not isinstance(entry, dict):
            return None
        try:
            start_index = int(entry.get("start_index"))
            end_index = int(entry.get("end_index"))
        except (TypeError, ValueError):
            return None
        if not (0 <= start_index < n and 0 <= end_index < n and start_index <= end_index):
            return None
        parts.append(
            (float(transcript_segments[start_index]["start"]), float(transcript_segments[end_index]["end"]))
        )

    for previous, following in zip(parts, parts[1:]):
        if following[0] < previous[1]:
            return None

    total = sum(end - start for start, end in parts)
    if total < min_len or total > max_len:
        return None
    return parts


def _combined_reason(item: dict) -> str:
    """Merge the response's `topic` (what is actually said) and `reason`
    (why it works as a clip) into the single human-readable string stored
    on candidate_segments.llm_reason and shown in the dev console.

    Combined rather than stored separately on purpose: `topic` exists to
    force the model to ground its pick in real content (see
    build_segment_suggestion_prompt), and a reviewer reading "why was this
    clip proposed?" wants both halves together. Keeping it one string also
    means the substance-focused prompt needed no schema/migration change
    at all. A response that omits `topic` (an older/smaller model ignoring
    part of the format) still yields a usable reason rather than being
    dropped -- the field is a prompt-level lever, not a hard contract.
    """
    topic = str(item.get("topic") or "").strip()[:MAX_TOPIC_CHARS]
    reason = str(item.get("reason") or "").strip()[:MAX_REASON_CHARS]
    if topic and reason:
        return f"{topic} -- {reason}"
    return topic or reason or _DEFAULT_REASON
