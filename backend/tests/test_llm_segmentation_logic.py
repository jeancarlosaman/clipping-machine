"""Unit tests for app.core.llm_segmentation_logic -- pure prompt building
and response parsing, no network involved (see test_llm_segmentation.py
for the OpenAI-calling half, mocked the same way as
test_caption_generation.py mocks app.core.caption_generation).
"""
import pytest

from app.core.llm_segmentation_logic import (
    build_feedback_examples_block,
    build_segment_suggestion_prompt,
    parse_segment_suggestions,
)

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "This is insane, wow!"},
    {"start": 2.5, "end": 4.0, "text": "Can you believe that?"},
    {"start": 4.1, "end": 6.0, "text": "That was hilarious and unreal."},
    {"start": 6.5, "end": 9.0, "text": "yeah okay sure whatever"},
]


def test_build_prompt_includes_every_segment_indexed_and_bounds():
    prompt = build_segment_suggestion_prompt(SEGMENTS, min_len=15.0, max_len=90.0, max_suggestions=8)
    assert "[0] 0.00-2.00: This is insane, wow!" in prompt
    assert "[3] 6.50-9.00: yeah okay sure whatever" in prompt
    assert "15-90 seconds" in prompt
    assert "up to 8" in prompt
    # The whole point of index-based prompting: never ask the LLM to emit
    # raw timestamps directly.
    assert "NEVER timestamps" in prompt


def test_parse_happy_path_computes_real_seconds_from_indices():
    raw = '[{"start_index": 0, "end_index": 2, "reason": "Big reaction into a punchline."}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result == [{"start": 0.0, "end": 6.0, "reason": "Big reaction into a punchline."}]


def test_parse_strips_markdown_fences():
    raw = '```json\n[{"start_index": 0, "end_index": 1, "reason": "ok"}]\n```'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result == [{"start": 0.0, "end": 4.0, "reason": "ok"}]


def test_parse_drops_out_of_range_index_but_keeps_other_valid_entries():
    raw = (
        '[{"start_index": 99, "end_index": 100, "reason": "bad"}, '
        '{"start_index": 1, "end_index": 2, "reason": "good"}]'
    )
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result == [{"start": 2.5, "end": 6.0, "reason": "good"}]


def test_parse_drops_end_index_before_start_index():
    raw = '[{"start_index": 2, "end_index": 0, "reason": "backwards"}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result == []


def test_parse_drops_window_shorter_than_min_len():
    # [0]-[0] is only 2.0s -- below a 15s floor, should be dropped entirely
    # rather than stretched.
    raw = '[{"start_index": 0, "end_index": 0, "reason": "too short"}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=15.0, max_len=90.0, max_suggestions=8)
    assert result == []


def test_parse_drops_window_longer_than_max_len():
    raw = '[{"start_index": 0, "end_index": 3, "reason": "too long"}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=5.0, max_suggestions=8)
    assert result == []


def test_parse_deduplicates_identical_windows():
    raw = (
        '[{"start_index": 0, "end_index": 1, "reason": "first"}, '
        '{"start_index": 0, "end_index": 1, "reason": "duplicate"}]'
    )
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert len(result) == 1
    assert result[0]["reason"] == "first"


def test_parse_caps_at_max_suggestions():
    import json

    items = [{"start_index": i, "end_index": i, "reason": f"r{i}"} for i in range(len(SEGMENTS))]
    raw = json.dumps(items)
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=0.5, max_len=90.0, max_suggestions=2)
    assert len(result) == 2


def test_parse_defaults_missing_reason():
    raw = '[{"start_index": 0, "end_index": 1}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result[0]["reason"] == "Suggested by LLM analysis of the transcript."


def test_parse_ignores_non_dict_elements():
    raw = '["not a dict", 42, {"start_index": 0, "end_index": 1, "reason": "ok"}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert len(result) == 1


def test_parse_raises_on_non_json():
    with pytest.raises(ValueError):
        parse_segment_suggestions("not json at all", SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)


def test_parse_raises_when_top_level_is_not_a_list():
    with pytest.raises(ValueError):
        parse_segment_suggestions('{"start_index": 0}', SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)


def test_parse_empty_array_returns_empty_list():
    assert parse_segment_suggestions("[]", SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8) == []


# ---- substance-focused prompt (topic field, content categories) ----


def test_build_prompt_asks_for_substance_not_just_energy():
    prompt = build_segment_suggestion_prompt(SEGMENTS, min_len=15.0, max_len=90.0, max_suggestions=8)
    # The specific failure this rewrite addresses: the old prompt judged
    # moments purely on delivery/affect and never asked what was said.
    assert "WHAT IS ACTUALLY SAID" in prompt
    assert "reasoning behind it" in prompt
    assert "setup and an actual payoff" in prompt
    # Short-form requirements the old prompt never stated at all.
    assert "self-contained" in prompt
    assert "start at the SETUP" in prompt
    assert "filler" in prompt


def test_build_prompt_requires_a_concrete_topic_field():
    prompt = build_segment_suggestion_prompt(SEGMENTS, min_len=15.0, max_len=90.0, max_suggestions=8)
    assert '"topic"' in prompt
    # The example in the prompt matters -- it's what stops the model
    # answering "a funny moment" for every pick.
    assert "NOT" in prompt and "a funny moment" in prompt


def test_parse_combines_topic_and_reason_into_stored_reason():
    raw = (
        '[{"start_index": 0, "end_index": 2, "topic": "explains why he refunded the sponsor", '
        '"reason": "Self-contained story with a clear payoff."}]'
    )
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result[0]["reason"] == (
        "explains why he refunded the sponsor -- Self-contained story with a clear payoff."
    )


def test_parse_still_accepts_a_response_without_a_topic():
    # A smaller model ignoring part of the format shouldn't lose the
    # suggestion entirely -- topic is a prompt-level lever, not a contract.
    raw = '[{"start_index": 0, "end_index": 2, "reason": "Big reaction into a punchline."}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result[0]["reason"] == "Big reaction into a punchline."


def test_parse_accepts_topic_only_response():
    raw = '[{"start_index": 0, "end_index": 2, "topic": "the sponsor refund story"}]'
    result = parse_segment_suggestions(raw, SEGMENTS, min_len=1.0, max_len=90.0, max_suggestions=8)
    assert result[0]["reason"] == "the sponsor refund story"


# ---- few-shot feedback examples ----


def test_feedback_block_is_empty_without_examples():
    assert build_feedback_examples_block(None) == ""
    assert build_feedback_examples_block([]) == ""


def test_feedback_block_renders_kept_and_cut_examples():
    block = build_feedback_examples_block(
        [
            {"decision": "approved", "notes": "the refund rant is the actual story", "transcript": "so I asked for a refund"},
            {"decision": "rejected", "notes": "starts mid-sentence, no context", "transcript": "...and that's why"},
        ]
    )
    assert "KEPT" in block and "CUT" in block
    assert "the refund rant is the actual story" in block
    assert "starts mid-sentence, no context" in block
    assert "so I asked for a refund" in block


def test_feedback_block_skips_examples_with_no_comment():
    # An approve/reject with no written reasoning teaches the model nothing
    # here -- it should be dropped, not rendered as a blank example.
    block = build_feedback_examples_block(
        [
            {"decision": "approved", "notes": "   ", "transcript": "some clip text"},
            {"decision": "approved", "notes": None, "transcript": "other clip text"},
        ]
    )
    assert block == ""


def test_feedback_block_handles_missing_transcript_excerpt():
    block = build_feedback_examples_block([{"decision": "approved", "notes": "great pacing"}])
    assert "great pacing" in block
    assert "clip said" not in block


def test_feedback_examples_appear_in_the_full_prompt():
    prompt = build_segment_suggestion_prompt(
        SEGMENTS,
        min_len=15.0,
        max_len=90.0,
        max_suggestions=8,
        feedback_examples=[
            {"decision": "approved", "notes": "loved the tangent about pricing", "transcript": "about pricing"}
        ],
    )
    assert "loved the tangent about pricing" in prompt
    assert "their taste overrides" in prompt.lower()


# ---- multi-part (stitched) suggestions ----

LONG_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "setting up the story"},
    {"start": 5.0, "end": 10.0, "text": "first half of the point"},
    {"start": 60.0, "end": 65.0, "text": "the callback much later"},
    {"start": 65.0, "end": 70.0, "text": "and the payoff"},
]


def test_build_prompt_offers_stitching_but_discourages_overuse():
    prompt = build_segment_suggestion_prompt(LONG_SEGMENTS, min_len=5.0, max_len=90.0, max_suggestions=8)
    assert '"parts"' in prompt
    assert "not next to each other" in prompt
    # Must be framed as the exception, not the default.
    assert "Most moments should NOT use this" in prompt


def test_parse_accepts_a_two_part_stitched_suggestion():
    raw = (
        '[{"start_index": 0, "end_index": 3, "topic": "the callback story", "reason": "Setup and payoff.", '
        '"parts": [{"start_index": 0, "end_index": 1}, {"start_index": 2, "end_index": 3}]}]'
    )
    result = parse_segment_suggestions(raw, LONG_SEGMENTS, min_len=5.0, max_len=90.0, max_suggestions=8)
    assert len(result) == 1
    assert result[0]["parts"] == [[0.0, 10.0], [60.0, 70.0]]
    # start/end describe the SPAN the clip is drawn from.
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 70.0


def test_parse_multipart_length_check_uses_summed_parts_not_the_span():
    # Parts total 20s; the span is 70s. A 30s max must accept this (20 <= 30)
    # rather than rejecting it on the span.
    raw = (
        '[{"start_index": 0, "end_index": 3, "reason": "ok", '
        '"parts": [{"start_index": 0, "end_index": 1}, {"start_index": 2, "end_index": 3}]}]'
    )
    result = parse_segment_suggestions(raw, LONG_SEGMENTS, min_len=5.0, max_len=30.0, max_suggestions=8)
    assert len(result) == 1
    assert result[0]["parts"] == [[0.0, 10.0], [60.0, 70.0]]


def test_parse_falls_back_to_single_range_when_parts_are_malformed():
    # A bad `parts` costs the suggestion its stitching, never the whole
    # suggestion -- start_index/end_index are present either way.
    raw = (
        '[{"start_index": 0, "end_index": 1, "reason": "still fine", '
        '"parts": [{"start_index": 99, "end_index": 100}, {"start_index": 2, "end_index": 3}]}]'
    )
    result = parse_segment_suggestions(raw, LONG_SEGMENTS, min_len=5.0, max_len=90.0, max_suggestions=8)
    assert len(result) == 1
    assert "parts" not in result[0]
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 10.0


def test_parse_rejects_overlapping_or_out_of_order_parts():
    raw = (
        '[{"start_index": 0, "end_index": 3, "reason": "backwards", '
        '"parts": [{"start_index": 2, "end_index": 3}, {"start_index": 0, "end_index": 1}]}]'
    )
    result = parse_segment_suggestions(raw, LONG_SEGMENTS, min_len=5.0, max_len=90.0, max_suggestions=8)
    # Falls back to the single-range interpretation rather than silently
    # re-sorting a sequence the model may have deliberately ordered.
    assert len(result) == 1
    assert "parts" not in result[0]


def test_parse_rejects_more_than_three_parts():
    raw = (
        '[{"start_index": 0, "end_index": 3, "reason": "montage", "parts": ['
        '{"start_index": 0, "end_index": 0}, {"start_index": 1, "end_index": 1}, '
        '{"start_index": 2, "end_index": 2}, {"start_index": 3, "end_index": 3}]}]'
    )
    result = parse_segment_suggestions(raw, LONG_SEGMENTS, min_len=5.0, max_len=90.0, max_suggestions=8)
    assert len(result) == 1
    assert "parts" not in result[0]


def test_parse_single_range_suggestion_has_no_parts_key():
    raw = '[{"start_index": 0, "end_index": 1, "reason": "plain"}]'
    result = parse_segment_suggestions(raw, LONG_SEGMENTS, min_len=5.0, max_len=90.0, max_suggestions=8)
    assert "parts" not in result[0]
