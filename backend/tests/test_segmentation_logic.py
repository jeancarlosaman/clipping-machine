"""Real (no mocking) tests for app.core.segmentation_logic.

detect_scene_cuts runs real PySceneDetect against a real generated video;
merge_into_speech_blocks and build_candidate_windows are pure functions
tested directly against constructed transcript-segment data.
"""
import pytest

from app.core.segmentation_logic import (
    build_candidate_windows,
    detect_scene_cuts,
    merge_into_speech_blocks,
)


def _segment(start, end, text="hello"):
    return {"start": start, "end": end, "text": text}


# ---- detect_scene_cuts ----


def test_detect_scene_cuts_finds_real_color_changes(three_scene_video_path):
    cuts = detect_scene_cuts(str(three_scene_video_path))
    assert len(cuts) == 2
    assert cuts[0] == pytest.approx(3.0, abs=0.2)
    assert cuts[1] == pytest.approx(6.0, abs=0.2)


def test_detect_scene_cuts_with_frame_skip_still_finds_cuts(three_scene_video_path):
    # frame_skip trades detection precision for decode speed (see the
    # function's own docstring) -- looser tolerance here on purpose: at
    # this fixture's 10fps, frame_skip=2 analyzes every 3rd frame (~0.3s
    # apart), so a detected cut can land up to ~0.3s after its true
    # boundary instead of within one frame of it.
    cuts = detect_scene_cuts(str(three_scene_video_path), frame_skip=2)
    assert len(cuts) == 2
    assert cuts[0] == pytest.approx(3.0, abs=0.5)
    assert cuts[1] == pytest.approx(6.0, abs=0.5)


# ---- merge_into_speech_blocks ----


def test_merge_empty_segments_returns_empty():
    assert merge_into_speech_blocks([]) == []


def test_merge_short_gaps_into_one_block():
    segments = [_segment(0.0, 2.0), _segment(2.3, 4.0), _segment(4.5, 6.0)]
    blocks = merge_into_speech_blocks(segments, max_gap_seconds=1.0)
    assert blocks == [(0.0, 6.0)]


def test_long_gap_splits_into_separate_blocks():
    segments = [_segment(0.0, 2.0), _segment(10.0, 12.0)]
    blocks = merge_into_speech_blocks(segments, max_gap_seconds=1.0)
    assert blocks == [(0.0, 2.0), (10.0, 12.0)]


def test_merge_handles_unsorted_input():
    segments = [_segment(10.0, 12.0), _segment(0.0, 2.0)]
    blocks = merge_into_speech_blocks(segments, max_gap_seconds=1.0)
    assert blocks == [(0.0, 2.0), (10.0, 12.0)]


def test_merge_handles_overlapping_segments():
    # Real transcripts shouldn't overlap, but a provider quirk producing
    # overlapping/out-of-order segment bounds shouldn't crash or lose time.
    segments = [_segment(0.0, 3.0), _segment(2.0, 5.0)]
    blocks = merge_into_speech_blocks(segments, max_gap_seconds=1.0)
    assert blocks == [(0.0, 5.0)]


# ---- build_candidate_windows ----


def test_in_range_block_becomes_one_window():
    segments = [_segment(0.0, 20.0)]
    windows = build_candidate_windows(segments, scene_cuts=[], min_len=15.0, max_len=90.0)
    assert windows == [(0.0, 20.0)]


def test_too_short_block_is_dropped():
    segments = [_segment(0.0, 5.0)]
    windows = build_candidate_windows(segments, scene_cuts=[], min_len=15.0, max_len=90.0)
    assert windows == []


def test_no_speech_at_all_returns_no_windows():
    assert build_candidate_windows([], scene_cuts=[1.0, 2.0]) == []


def test_long_block_produces_overlapping_windows_snapped_to_real_scene_cuts(three_scene_video_path):
    # One continuous 9s speech block over a video with real cuts at 3.0/6.0
    # (from the fixture) -- forcing max_len=4.0 should slide across the
    # block (see _sliding_subwindows) rather than committing to one rigid
    # partition, with each window's end snapped to a nearby real scene cut
    # rather than an arbitrary 4s/8s mark.
    scene_cuts = detect_scene_cuts(str(three_scene_video_path))
    segments = [_segment(0.0, 9.0)]

    windows = build_candidate_windows(
        segments, scene_cuts, min_len=2.0, max_len=4.0, scene_search_window=2.0
    )

    # More than one candidate covering the block -- this is the whole point
    # of sliding over it instead of a single fixed partition.
    assert len(windows) > 1
    # Every window is inside the block and no shorter than min_len.
    for start, end in windows:
        assert 0.0 <= start < end <= 9.0
        assert end - start >= 2.0
    # The block's start and end are both covered by some window.
    assert any(start == pytest.approx(0.0, abs=0.01) for start, _ in windows)
    assert any(end == pytest.approx(9.0, abs=0.01) for _, end in windows)
    # At least one window boundary actually snapped to a real detected cut
    # (3.0 or 6.0) rather than landing on a hard-cut multiple of the step.
    boundaries = [b for start, end in windows for b in (start, end)]
    assert any(b == pytest.approx(3.0, abs=0.2) for b in boundaries)
    assert any(b == pytest.approx(6.0, abs=0.2) for b in boundaries)


def test_long_block_hard_cuts_when_no_scene_cut_nearby():
    segments = [_segment(0.0, 9.0)]
    # No scene cuts at all -- every window boundary must fall back to a hard
    # cut at exactly the target length, never raise, never loop forever.
    windows = build_candidate_windows(segments, scene_cuts=[], min_len=2.0, max_len=4.0)
    assert windows == [(0.0, 4.0), (2.0, 6.0), (4.0, 8.0)]
    for start, end in windows:
        assert end - start == pytest.approx(4.0)


def test_sliding_subwindows_bounded_for_pathologically_long_block():
    # A single unbroken 40-minute "block" (no silence gaps at all) must not
    # generate an unbounded number of candidates -- see
    # _MAX_SUBWINDOWS_PER_BLOCK in app.core.segmentation_logic.
    segments = [_segment(0.0, 2400.0)]
    windows = build_candidate_windows(segments, scene_cuts=[], min_len=15.0, max_len=90.0)
    assert 0 < len(windows) <= 61  # cap + the always-appended tail window
