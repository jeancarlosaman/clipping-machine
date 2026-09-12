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
        f"Pick the {max_suggestions} best moments in this transcript to cut as standalone "
        "short-form clips (TikTok / Reels / Shorts). You are choosing what a viewer scrolling "
        "their feed will stop on and watch to the end.\n\n"

        "WHAT ACTUALLY PERFORMS IN SHORT FORM\n"
        "The one thing every short-form platform has publicly confirmed it weights heavily is "
        "whether people watch a video all the way through. Everything below serves that:\n"
        "  - THE HOOK. The first 2-3 seconds decide whether anyone sees the rest. The clip must "
        "open on something that creates a question, a stake, or a reason to stay -- a bold claim, "
        "a strong reaction, the setup of a story, an argument starting. It must NOT open on "
        "throat-clearing, a greeting, or slow preamble.\n"
        "  - A PAYOFF. Something has to actually resolve or land before the clip ends. A moment "
        "that builds and then trails off is the most common way a clip dies.\n"
        "  - TENSION OR STAKES. Disagreement, being proven wrong, a risky opinion, a near-miss, "
        "something going badly, a genuine surprise.\n"
        "  - A REASON TO CARE. Something specific and concrete: a number, a name, a real "
        "consequence, information the viewer did not have.\n"
        "  - IT MUST BE UNDERSTANDABLE COLD, with no knowledge of the rest of the VOD.\n\n"

        "Use your own judgement about what tends to travel in short form. Strong material "
        "usually looks like:\n"
        "  - a surprising claim, fact, number or detail worth knowing\n"
        "  - a strong opinion WITH the reasoning behind it\n"
        "  - a story with a real setup and a real payoff\n"
        "  - a clear explanation of something non-obvious that people get wrong\n"
        "  - a genuine disagreement, correction, or visible change of mind\n"
        "  - a joke, big reaction or dramatic reveal -- but only when the CONTENT behind it "
        "lands, not just because the reaction is loud\n\n"

        "Reject, however energetic it sounds: filler, small talk, greetings, stream logistics, "
        "reading chat aloud, and anything that only makes sense if you watched the last hour.\n\n"

        "STITCHING SEPARATE PARTS TOGETHER\n"
        "You do NOT have to pick one continuous stretch. You can build a single clip out of 2 or "
        "3 separate pieces of the VOD, joined in playback order, using the \"parts\" field. This "
        "is a genuinely useful tool -- reach for it whenever it produces a better clip than any "
        "single stretch would, for example:\n"
        "  - the setup happens well before the payoff, with unrelated material in between\n"
        "  - a claim early on is called back to, contradicted, or proven later\n"
        "  - the interesting exchange is interrupted by something irrelevant that should be cut\n"
        "  - a short punchy hook exists somewhere that would open the clip better than its own "
        "natural start\n"
        "Each part must end on a complete thought -- the viewer sees a hard cut, so a part ending "
        "mid-sentence sounds broken. Order the parts however the clip should play; they do not "
        "have to be in transcript order if a later line makes the stronger opening.\n\n"

        "HARD RULES -- a suggestion breaking any of these is discarded automatically:\n"
        "  - Reference the bracketed [index] numbers above. NEVER timestamps or seconds.\n"
        f"  - Every clip's total length MUST be between {min_len:.0f} and {max_len:.0f} seconds. "
        "Work it out from the segment times shown. This is enforced in code, not a preference -- "
        "a clip outside this range is thrown away, so extend or trim your index range to fit. "
        "For a stitched clip this applies to the COMBINED length of its parts.\n"
        "  - No two suggestions may cover substantially the same moment.\n"
        "  - Spread your picks across the whole VOD. Do not take several clips out of one short stretch just because it was lively -- a later, calmer moment with a real point beats a third clip from the same two minutes.\n\n"

        "Reply with ONLY a JSON array (no markdown fences, no other text). Each element:\n"
        '  "start_index": <int, index from the list above>\n'
        '  "end_index": <int, index from the list above, >= start_index>\n'
        '  "hook": the specific thing in the opening seconds that stops a scroll, quoted or '
        'described in a few words\n'
        '  "topic": what is actually said, stated concretely in one short phrase '
        '(e.g. "explains why he refunded the sponsor" -- NOT "a funny moment")\n'
        '  "reason": one sentence on why this holds attention to the end\n'
        '  "strength": <int 1-5> how strong a clip you believe this is, 5 being the best in this '
        "VOD. Use the full range; do not mark everything 4 or 5.\n"
        '  "parts": OPTIONAL -- for a stitched clip, a list of 2-3 objects each with their own '
        '"start_index" and "end_index", in playback order. Omit entirely for a single continuous '
        "clip. When present, still set start_index/end_index to the first part's start and the "
        "last part's end.\n\n"

        f"Return your best {max_suggestions} even if some are stronger than others -- a weaker "
        "suggestion marked strength 2 is more useful than no suggestion. Only return an empty "
        "array if the transcript is genuinely unusable (silence, or nothing but logistics)."
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
    dropped: list[str] | None = None,
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

    Pass a list as `dropped` to find out WHY things were skipped. Without it
    a response where every suggestion failed validation is indistinguishable
    from a model that returned an empty array -- both just log "count: 0",
    which is exactly the dead end this argument exists to remove.
    """
    if dropped is None:
        dropped = []
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
            dropped.append("not_an_object")
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
            dropped.append("missing_or_non_integer_indices")
            continue
        if not (0 <= start_index < n and 0 <= end_index < n and start_index <= end_index):
            dropped.append(f"index_out_of_range({start_index},{end_index} of {n})")
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
            dropped.append(f"duration_{duration:.1f}s_outside_{min_len:.0f}-{max_len:.0f}s")
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
    """The human-readable explanation stored on the candidate row.

    Merges the model's fields into one string because they share a single
    column (candidate_segments.llm_reason): `topic` says WHAT the moment is,
    `hook` why its opening stops a scroll, `strength` the model's own ranking.
    Any of them missing simply drops out, so a smaller model returning only
    `reason` behaves exactly as before.
    """
    topic = str(item.get("topic") or "").strip()[:MAX_TOPIC_CHARS]
    hook = str(item.get("hook") or "").strip()[:MAX_TOPIC_CHARS]
    reason = str(item.get("reason") or "").strip()
    strength = parse_strength(item)

    bits = []
    if strength:
        bits.append(f"[{strength}/5]")
    if topic:
        bits.append(topic)
    if hook:
        bits.append(f"Hook: {hook}")
    if reason:
        bits.append(reason)
    return " -- ".join(bits) if bits else ""


def parse_strength(item: dict) -> int | None:
    """The model's own 1-5 rating for a suggestion, or None if absent/invalid.

    Used to decide which suggestions survive the max_suggestions cap, so the
    model's best picks get through rather than whichever came first in the
    response.
    """
    try:
        value = int(item.get("strength"))
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 5 else None
