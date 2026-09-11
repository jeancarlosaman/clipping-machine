"""Pure unit tests for app.core.scoring_logic -- no DB/IO needed, per the
same rationale as test_segmentation_logic.py.
"""
import pytest

from app.core.scoring_logic import (
    HOOK_WINDOW_SECONDS,
    ScoreWeights,
    default_caption,
    emotional_language_hits,
    hook_strength,
    pause_count,
    question_mark_count,
    scene_change_count,
    score_multipart_window,
    score_out_of_10,
    score_window,
    select_top_non_overlapping,
    speech_density,
    window_text,
)

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "This is insane, wow!"},
    {"start": 2.5, "end": 4.0, "text": "Can you believe that?"},
    {"start": 4.1, "end": 6.0, "text": "That was hilarious and unreal."},
]


def test_window_text_only_includes_overlapping_segments():
    assert window_text(SEGMENTS, 0.0, 4.0) == "This is insane, wow! Can you believe that?"


def test_window_text_partial_overlap_still_included():
    # A segment overlapping the window boundary at all should be included,
    # even if it extends past the window edge.
    text = window_text(SEGMENTS, 5.0, 10.0)
    assert "hilarious" in text


def test_speech_density_zero_duration_is_zero():
    assert speech_density(SEGMENTS, 5.0, 5.0) == 0.0


def test_speech_density_counts_words_over_duration():
    density = speech_density(SEGMENTS, 0.0, 2.0)
    assert density == 4 / 2.0  # "This is insane wow" -> 4 words


def test_pause_count_detects_beat_gaps():
    # Gap between seg0 end (2.0) and seg1 start (2.5) is 0.5s >= threshold;
    # gap between seg1 end (4.0) and seg2 start (4.1) is 0.1s < threshold.
    assert pause_count(SEGMENTS, 0.0, 6.0) == 1


def test_question_mark_count():
    assert question_mark_count(SEGMENTS, 0.0, 6.0) == 1


def test_emotional_language_hits_counts_lexicon_words():
    # "insane", "wow", "hilarious", "unreal" == 4 hits.
    assert emotional_language_hits(SEGMENTS, 0.0, 6.0) == 4


def test_emotional_language_hits_word_boundary_not_substring():
    segments = [{"start": 0.0, "end": 1.0, "text": "I wowed the crowd"}]
    assert emotional_language_hits(segments, 0.0, 1.0) == 0


def test_scene_change_count_only_counts_inside_window():
    cuts = [1.0, 3.0, 10.0]
    assert scene_change_count(cuts, 0.0, 5.0) == 2


# ---- hook_strength ----


def test_hook_strength_immediate_plain_speech_scores_immediacy_only():
    segments = [{"start": 0.0, "end": 3.0, "text": "just some normal talking here"}]
    # No dead air (speech starts at the window's own start), no question
    # mark/emotional-language hit -- only the immediacy component applies.
    assert hook_strength(segments, 0.0, 10.0) == 0.7


def test_hook_strength_immediate_speech_with_question_hits_full_score():
    segments = [{"start": 0.0, "end": 3.0, "text": "can you believe this?"}]
    assert hook_strength(segments, 0.0, 10.0) == 1.0


def test_hook_strength_immediate_speech_with_emotional_language_hits_full_score():
    segments = [{"start": 0.0, "end": 3.0, "text": "this is absolutely insane"}]
    assert hook_strength(segments, 0.0, 10.0) == 1.0


def test_hook_strength_no_speech_in_hook_window_is_zero():
    # Speech exists in the window overall, just not within the opening
    # HOOK_WINDOW_SECONDS -- total dead air for the hook itself.
    segments = [{"start": 5.0, "end": 6.0, "text": "finally something happens"}]
    assert hook_strength(segments, 0.0, 10.0) == 0.0


def test_hook_strength_partial_dead_air_scales_immediacy_linearly():
    # Speech starts halfway through the 3s hook window -> half credit on
    # the immediacy component, no attention bonus.
    segments = [{"start": 1.5, "end": 3.0, "text": "hello there everyone"}]
    assert hook_strength(segments, 0.0, 10.0, hook_window_seconds=3.0) == 0.35


def test_hook_strength_zero_duration_hook_window_is_zero():
    assert hook_strength(SEGMENTS, 5.0, 5.0) == 0.0


def test_hook_strength_respects_custom_hook_window_seconds():
    # Speech starting at 4.0 is total dead air under the default 3s window
    # (outside it entirely) but partially inside a wider 5s one -- 1s of
    # immediacy credit out of 5s, i.e. 0.7 * (1 - 4/5) = 0.14.
    segments = [{"start": 4.0, "end": 5.0, "text": "hey everyone"}]
    assert hook_strength(segments, 0.0, 10.0, hook_window_seconds=3.0) == 0.0
    assert hook_strength(segments, 0.0, 10.0, hook_window_seconds=5.0) == pytest.approx(0.14)


def test_hook_window_seconds_default_matches_constant():
    assert HOOK_WINDOW_SECONDS == 3.0


def test_score_window_zero_weights_gives_zero_composite():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0)
    composite, breakdown = score_window(SEGMENTS, 0.0, 6.0, [], weights)
    assert composite == 0.0
    assert breakdown["composite"] == 0.0


def test_score_window_breakdown_has_expected_shape():
    weights = ScoreWeights(
        speech_density=1.0, pause_count=0.5, question_marks=1.0,
        emotional_language=1.5, scene_changes=0.5, motion_proxy=0.0,
    )
    composite, breakdown = score_window(SEGMENTS, 0.0, 6.0, [3.0], weights)
    assert composite > 0
    for key in ("raw", "normalized", "weights", "contributions", "composite"):
        assert key in breakdown
    for feature in (
        "speech_density", "pause_count", "question_marks", "emotional_language",
        "scene_changes", "motion_proxy", "hook_strength",
    ):
        assert feature in breakdown["raw"]
        assert 0.0 <= breakdown["normalized"][feature] <= 1.0


def test_score_window_higher_signal_window_scores_higher():
    weights = ScoreWeights(
        speech_density=1.0, pause_count=0.5, question_marks=1.0,
        emotional_language=1.5, scene_changes=0.5, motion_proxy=0.0,
    )
    quiet = [{"start": 0.0, "end": 10.0, "text": "yeah okay sure"}]
    exciting = SEGMENTS
    quiet_score, _ = score_window(quiet, 0.0, 10.0, [], weights)
    exciting_score, _ = score_window(exciting, 0.0, 6.0, [1.0, 3.0], weights)
    assert exciting_score > quiet_score


def test_score_window_without_audio_events_treats_laughter_as_zero():
    # No `audio_events` passed -- the pre-existing call shape, and the
    # behavior every caller got before this feature existed. Confirms
    # adding the feature didn't change scoring for anyone not opted in.
    weights = ScoreWeights(0, 0, 0, 0, 0, 0, laughter=2.0, crowd_reaction=2.0)
    composite, breakdown = score_window(SEGMENTS, 0.0, 6.0, [], weights)
    assert composite == 0.0
    assert breakdown["raw"]["laughter"] == 0.0
    assert breakdown["raw"]["crowd_reaction"] == 0.0


def test_score_window_with_audio_events_contributes_to_composite():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0, laughter=2.0, crowd_reaction=1.0)
    composite, breakdown = score_window(
        SEGMENTS, 0.0, 6.0, [], weights, audio_events={"laughter": 0.8, "crowd_reaction": 0.5}
    )
    assert breakdown["raw"]["laughter"] == 0.8
    assert breakdown["raw"]["crowd_reaction"] == 0.5
    # laughter=1.0 cap -- already a 0..1 probability, no rescaling -- so
    # contribution is value * weight directly.
    assert breakdown["contributions"]["laughter"] == 0.8 * 2.0
    assert breakdown["contributions"]["crowd_reaction"] == 0.5 * 1.0
    assert composite == breakdown["contributions"]["laughter"] + breakdown["contributions"]["crowd_reaction"]


def test_score_window_audio_events_missing_key_defaults_to_zero():
    # A partial dict (e.g. detector only reports one signal) shouldn't crash.
    weights = ScoreWeights(0, 0, 0, 0, 0, 0, laughter=1.0, crowd_reaction=1.0)
    _, breakdown = score_window(SEGMENTS, 0.0, 6.0, [], weights, audio_events={"laughter": 0.3})
    assert breakdown["raw"]["laughter"] == 0.3
    assert breakdown["raw"]["crowd_reaction"] == 0.0


def test_score_window_hook_strength_defaults_to_zero_weight():
    # Matches laughter/crowd_reaction's convention: ScoreWeights' own field
    # default is 0.0 so "everything off" positional construction still
    # means everything off -- hook_strength is turned on by default at the
    # config layer (settings.score_weight_hook_strength), not here.
    weights = ScoreWeights(0, 0, 0, 0, 0, 0)
    segments = [{"start": 0.0, "end": 1.0, "text": "can you believe it?"}]
    composite, breakdown = score_window(segments, 0.0, 6.0, [], weights)
    assert breakdown["raw"]["hook_strength"] == 1.0  # feature itself still computed
    assert breakdown["contributions"]["hook_strength"] == 0.0  # but weighted at zero
    assert composite == 0.0


def test_score_window_hook_strength_contributes_when_weighted():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0, hook_strength=2.0)
    segments = [{"start": 0.0, "end": 1.0, "text": "just talking, nothing special"}]
    composite, breakdown = score_window(segments, 0.0, 6.0, [], weights)
    assert breakdown["raw"]["hook_strength"] == 0.7  # immediate speech, no attention bonus
    assert breakdown["contributions"]["hook_strength"] == 0.7 * 2.0
    assert composite == breakdown["contributions"]["hook_strength"]


def test_default_caption_short_text_returned_verbatim():
    assert default_caption(SEGMENTS, 0.0, 2.0) == "This is insane, wow!"


def test_default_caption_truncates_on_word_boundary():
    segments = [{"start": 0.0, "end": 1.0, "text": "word " * 50}]
    caption = default_caption(segments, 0.0, 1.0, max_chars=20)
    assert len(caption) <= 24  # 20 + "..." plus a little slack
    assert caption.endswith("...")
    assert not caption[:-3].endswith(" ")


def test_default_caption_empty_window_is_empty_string():
    assert default_caption(SEGMENTS, 100.0, 200.0) == ""


# ---- score_out_of_10 ----


def test_score_out_of_10_bounded_zero_to_ten():
    weights = ScoreWeights(
        speech_density=1.0, pause_count=0.5, question_marks=1.0,
        emotional_language=1.5, scene_changes=0.5, motion_proxy=0.0,
    )
    _, breakdown = score_window(SEGMENTS, 0.0, 6.0, [1.0, 3.0], weights)
    assert 0.0 <= breakdown["score_out_of_10"] <= 10.0
    # score_window embeds it in the breakdown; score_out_of_10() standalone
    # on that same breakdown must agree.
    assert score_out_of_10(breakdown) == breakdown["score_out_of_10"]


def test_score_out_of_10_stable_across_weight_configs():
    # The whole point of rescaling by sum(weights) is that a reviewer's
    # sense of "what's a good score" doesn't silently shift just because
    # SCORE_WEIGHT_* settings changed -- doubling every weight uniformly
    # must not change the displayed 0..10 score.
    base_weights = ScoreWeights(
        speech_density=1.0, pause_count=0.5, question_marks=1.0,
        emotional_language=1.5, scene_changes=0.5, motion_proxy=0.0,
    )
    doubled_weights = ScoreWeights(
        speech_density=2.0, pause_count=1.0, question_marks=2.0,
        emotional_language=3.0, scene_changes=1.0, motion_proxy=0.0,
    )
    _, base_breakdown = score_window(SEGMENTS, 0.0, 6.0, [1.0, 3.0], base_weights)
    _, doubled_breakdown = score_window(SEGMENTS, 0.0, 6.0, [1.0, 3.0], doubled_weights)
    assert base_breakdown["score_out_of_10"] == doubled_breakdown["score_out_of_10"]


def test_score_out_of_10_all_zero_weights_is_zero_not_a_crash():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0)
    _, breakdown = score_window(SEGMENTS, 0.0, 6.0, [], weights)
    assert breakdown["score_out_of_10"] == 0.0


# ---- select_top_non_overlapping ----


def test_select_top_non_overlapping_picks_highest_scores_when_disjoint():
    candidates = [
        {"id": "a", "start": 0.0, "end": 10.0, "score": 3.0},
        {"id": "b", "start": 20.0, "end": 30.0, "score": 8.0},
        {"id": "c", "start": 40.0, "end": 50.0, "score": 5.0},
    ]
    selected = select_top_non_overlapping(candidates, max_count=2)
    assert selected == {"b", "c"}


def test_select_top_non_overlapping_suppresses_overlapping_lower_score():
    # b and c both cover roughly the same moment as the top-scoring
    # candidate (a) -- without NMS a naive top-3 would return all three
    # near-duplicates instead of one clean pick plus a genuinely distinct one.
    candidates = [
        {"id": "a", "start": 0.0, "end": 10.0, "score": 9.0},
        {"id": "b", "start": 1.0, "end": 11.0, "score": 8.5},  # heavy overlap with a
        {"id": "c", "start": 2.0, "end": 9.0, "score": 8.0},   # heavy overlap with a
        {"id": "d", "start": 50.0, "end": 60.0, "score": 4.0},  # distinct
    ]
    selected = select_top_non_overlapping(candidates, max_count=3, overlap_threshold=0.5)
    assert "a" in selected
    assert "b" not in selected
    assert "c" not in selected
    assert "d" in selected
    assert len(selected) == 2  # only 2 genuinely distinct candidates exist


def test_select_top_non_overlapping_respects_min_score():
    candidates = [
        {"id": "a", "start": 0.0, "end": 10.0, "score": 9.0},
        {"id": "b", "start": 20.0, "end": 30.0, "score": 2.0},
    ]
    selected = select_top_non_overlapping(candidates, max_count=5, min_score=5.0)
    assert selected == {"a"}


def test_select_top_non_overlapping_empty_input():
    assert select_top_non_overlapping([], max_count=5) == set()


# ---- multi-part (stitched) scoring ----

MULTIPART_SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "This is insane, wow! Can you believe that?"},
    {"start": 50.0, "end": 54.0, "text": "That was hilarious and unreal."},
    # Material that exists only in the gap between the two parts -- must
    # never be counted, since the viewer never sees it.
    {"start": 20.0, "end": 24.0, "text": "unbelievable shocking crazy insane wow"},
]


def test_score_multipart_ignores_material_in_the_gap_between_parts():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0)
    parts = [(0.0, 4.0), (50.0, 54.0)]
    _, breakdown = score_multipart_window(MULTIPART_SEGMENTS, parts, [], weights)
    # The gap segment alone carries 5 lexicon hits; only the two parts'
    # own text should be counted.
    gap_hits = emotional_language_hits(MULTIPART_SEGMENTS, 20.0, 24.0)
    assert gap_hits > 0
    part_hits = emotional_language_hits(MULTIPART_SEGMENTS, 0.0, 4.0) + emotional_language_hits(
        MULTIPART_SEGMENTS, 50.0, 54.0
    )
    assert breakdown["raw"]["emotional_language"] == float(part_hits)


def test_score_multipart_speech_density_uses_playing_time_not_span():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0)
    parts = [(0.0, 4.0), (50.0, 54.0)]
    _, breakdown = score_multipart_window(MULTIPART_SEGMENTS, parts, [], weights)
    # 8s of real playing time, not the 54s span -- a span-based density
    # would divide the same words by nearly seven times the duration.
    words = len(
        (window_text(MULTIPART_SEGMENTS, 0.0, 4.0) + " " + window_text(MULTIPART_SEGMENTS, 50.0, 54.0)).split()
    )
    assert breakdown["raw"]["speech_density"] == pytest.approx(words / 8.0, rel=0.01)


def test_score_multipart_scene_changes_only_count_inside_parts():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0)
    parts = [(0.0, 4.0), (50.0, 54.0)]
    cuts = [2.0, 30.0, 52.0]  # 30.0 falls in the removed gap
    _, breakdown = score_multipart_window(MULTIPART_SEGMENTS, parts, cuts, weights)
    assert breakdown["raw"]["scene_changes"] == 2.0


def test_score_multipart_hook_strength_comes_from_the_first_part():
    weights = ScoreWeights(0, 0, 0, 0, 0, 0)
    parts = [(0.0, 4.0), (50.0, 54.0)]
    _, breakdown = score_multipart_window(MULTIPART_SEGMENTS, parts, [], weights)
    assert breakdown["raw"]["hook_strength"] == hook_strength(MULTIPART_SEGMENTS, 0.0, 4.0)


def test_score_multipart_breakdown_matches_single_window_shape():
    # A reviewer comparing a stitched clip's 0..10 score to a normal one has
    # to be comparing numbers produced the same way.
    weights = ScoreWeights(
        speech_density=1.0, pause_count=0.5, question_marks=1.0,
        emotional_language=1.5, scene_changes=0.5, motion_proxy=0.0,
    )
    _, single = score_window(MULTIPART_SEGMENTS, 0.0, 4.0, [], weights)
    _, multi = score_multipart_window(MULTIPART_SEGMENTS, [(0.0, 4.0), (50.0, 54.0)], [], weights)
    assert set(multi["raw"]) == set(single["raw"])
    assert set(multi["weights"]) == set(single["weights"])
    assert 0.0 <= multi["score_out_of_10"] <= 10.0
