"""Pure unit tests for app.core.caption_logic -- no network/DB/IO needed,
same rationale as test_scoring_logic.py / test_rendering_logic.py.
"""
import pytest

from app.core.caption_logic import (
    build_caption_prompt,
    heuristic_explanation,
    heuristic_hashtags,
    heuristic_title,
    parse_caption_response,
)

BREAKDOWN = {
    "contributions": {
        "speech_density": 0.3,
        "pause_count": 0.1,
        "question_marks": 0.0,
        "emotional_language": 1.2,
        "scene_changes": 0.4,
        "motion_proxy": 0.0,
    }
}


def test_heuristic_explanation_names_top_contributing_signals():
    text = heuristic_explanation(BREAKDOWN)
    assert "emotional" in text.lower()
    assert "scene change" in text.lower()


def test_heuristic_explanation_handles_none_breakdown():
    text = heuristic_explanation(None)
    assert text  # never empty


def test_heuristic_explanation_handles_all_zero_contributions():
    text = heuristic_explanation({"contributions": {"speech_density": 0.0}})
    assert "no single signal" in text.lower()


def test_heuristic_hashtags_always_returns_base_set():
    tags = heuristic_hashtags(None)
    assert len(tags) >= 3
    assert all(t.startswith("#") for t in tags)


def test_heuristic_hashtags_adds_signal_specific_tag():
    tags = heuristic_hashtags(BREAKDOWN)
    # top contribution is emotional_language -> should add its extra tag
    assert "#omg" in tags


def test_heuristic_title_picks_top_signal_template():
    # top contribution is emotional_language -> its specific hook template
    title = heuristic_title(BREAKDOWN)
    assert title == "You Won't Believe His Reaction"


def test_heuristic_title_handles_none_breakdown():
    title = heuristic_title(None)
    assert title  # never empty -- falls back to the default hook


def test_heuristic_title_handles_unknown_top_signal():
    title = heuristic_title({"contributions": {"some_future_feature": 5.0}})
    assert title == "You Have To See This Clip"  # default template


def test_build_caption_prompt_includes_transcript_and_hint():
    prompt = build_caption_prompt("this is insane wow", "Selected mainly for high emotional language.", "fallback")
    assert "this is insane wow" in prompt
    assert "Selected mainly for high emotional language." in prompt
    assert "hashtags" in prompt.lower()
    assert "title" in prompt.lower()


def test_build_caption_prompt_handles_empty_transcript():
    prompt = build_caption_prompt("", "hint", "fallback")
    assert "no speech detected" in prompt.lower()


def test_parse_caption_response_happy_path():
    raw = (
        '{"title": "He Clutched It!", "hashtags": ["insane play", "#Clutch"], '
        '"caption": "He clutched it!", "explanation": "Big win moment."}'
    )
    parsed = parse_caption_response(raw)
    assert parsed["title"] == "He Clutched It!"
    assert parsed["hashtags"] == ["#insaneplay", "#Clutch"]
    assert parsed["caption"] == "He clutched it!"
    assert parsed["explanation"] == "Big win moment."


def test_parse_caption_response_strips_markdown_fences():
    raw = '```json\n{"title": "Wow", "hashtags": ["clip"], "caption": "wow", "explanation": "x"}\n```'
    parsed = parse_caption_response(raw)
    assert parsed["hashtags"] == ["#clip"]


def test_parse_caption_response_dedupes_and_caps_hashtags():
    raw = (
        '{"title": "x", "hashtags": ["a", "a", "b", "c", "d", "e", "f", "g"], '
        '"caption": "x", "explanation": "y"}'
    )
    parsed = parse_caption_response(raw)
    assert parsed["hashtags"] == ["#a", "#b", "#c", "#d", "#e", "#f"]


def test_parse_caption_response_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_caption_response("not json at all")


def test_parse_caption_response_rejects_missing_hashtags():
    with pytest.raises(ValueError):
        parse_caption_response('{"title": "x", "caption": "x", "explanation": "y"}')


def test_parse_caption_response_rejects_missing_title():
    with pytest.raises(ValueError):
        parse_caption_response('{"hashtags": ["x"], "caption": "x", "explanation": "y"}')


def test_parse_caption_response_rejects_empty_title():
    with pytest.raises(ValueError):
        parse_caption_response('{"title": "  ", "hashtags": ["x"], "caption": "x", "explanation": "y"}')


def test_parse_caption_response_rejects_empty_caption():
    with pytest.raises(ValueError):
        parse_caption_response('{"title": "x", "hashtags": ["x"], "caption": "", "explanation": "y"}')


def test_parse_caption_response_clamps_length():
    long_caption = "x" * 500
    long_title = "y" * 200
    raw = (
        f'{{"title": "{long_title}", "hashtags": ["a"], "caption": "{long_caption}", "explanation": "y"}}'
    )
    parsed = parse_caption_response(raw)
    assert len(parsed["caption"]) == 150
    assert len(parsed["title"]) == 70
