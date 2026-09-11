"""Pure unit tests for app.core.rendering_logic -- no ffmpeg/DB/IO needed,
same rationale as test_scoring_logic.py / test_segmentation_logic.py.
"""
import pytest

from app.core.rendering_logic import (
    append_subtitles_stage,
    build_fit_frame_filtergraph,
    build_single_crop_filtergraph,
    build_split_reaction_filtergraph,
    build_concat_prefix,
    build_multipart_srt,
    build_srt,
    build_title_srt,
    classify_reaction_layout,
    crop_from_facecam_rect,
    compute_crop_offset,
    compute_face_zoom_crop,
    compute_vertical_crop,
    ffmpeg_subtitles_filter_path,
    normalize_parts,
    parts_duration,
    retarget_source_label,
)


def test_compute_vertical_crop_wide_source_crops_width():
    # 1920x1080 (16:9) -> crop width down to a 9:16 slice, full height.
    crop_w, crop_h = compute_vertical_crop(1920, 1080)
    assert crop_h == 1080
    assert crop_w == 606  # 1080 * 9/16 = 607.5, floored to the nearest even
    assert crop_w % 2 == 0


def test_compute_vertical_crop_exactly_target_ratio_crops_neither_dimension():
    # 1080x1920 is already exactly 9:16.
    crop_w, crop_h = compute_vertical_crop(1080, 1920)
    assert crop_w == 1080
    assert crop_h == 1920


def test_compute_vertical_crop_square_source_crops_width():
    # 1000x1000 (1:1) is wider than 9:16 -- width needs cropping down, not height.
    crop_w, crop_h = compute_vertical_crop(1000, 1000)
    assert crop_h == 1000
    assert crop_w < 1000
    assert crop_w % 2 == 0


def test_compute_vertical_crop_dimensions_always_even():
    # Odd source dims are common (e.g. 641x361) -- output must still be even
    # for libx264/yuv420p, or the encode fails outright.
    crop_w, crop_h = compute_vertical_crop(641, 361)
    assert crop_w % 2 == 0
    assert crop_h % 2 == 0


def test_compute_vertical_crop_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        compute_vertical_crop(0, 100)
    with pytest.raises(ValueError):
        compute_vertical_crop(100, -5)


def test_compute_crop_offset_no_focal_point_centers_like_before():
    # 1920x1080 source, 606x1080 crop (see the wide-source test above) --
    # with no focal point this must match ffmpeg's own auto-centering,
    # i.e. the original pre-face-crop behavior.
    x, y = compute_crop_offset(1920, 1080, 606, 1080, focal_point=None)
    assert (x, y) == ((1920 - 606) // 2, 0)


def test_compute_crop_offset_centers_on_focal_point():
    # Face detected near the left third of a 1920x1080 frame -- the crop
    # window should shift left (smaller x) to follow it instead of staying
    # centered on the frame.
    centered_x, _ = compute_crop_offset(1920, 1080, 606, 1080, focal_point=None)
    x, y = compute_crop_offset(1920, 1080, 606, 1080, focal_point=(400.0, 540.0))
    assert x < centered_x
    assert x == int(400 - 606 / 2)
    assert y == 0  # full height already kept -- nothing to offset vertically


def test_compute_crop_offset_clamps_to_left_edge():
    # Focal point near the very left edge -- crop window would run off the
    # frame if centered exactly on it; must clamp to x=0 instead.
    x, y = compute_crop_offset(1920, 1080, 606, 1080, focal_point=(10.0, 540.0))
    assert x == 0


def test_compute_crop_offset_clamps_to_right_edge():
    x, y = compute_crop_offset(1920, 1080, 606, 1080, focal_point=(1900.0, 540.0))
    assert x == 1920 - 606


def test_compute_crop_offset_no_room_on_uncropped_axis_ignores_focal_y():
    # crop_h == height (nothing to crop vertically) -- y must stay 0
    # regardless of what the focal point's y is.
    x, y = compute_crop_offset(1920, 1080, 606, 1080, focal_point=(400.0, 999.0))
    assert y == 0


# ---- classify_reaction_layout ----


def test_classify_reaction_layout_true_for_small_cornered_face():
    # A small (well under 18% of a 1920x1080 frame) face sitting in the
    # bottom-right corner -- the classic webcam-box placement.
    is_reaction = classify_reaction_layout((1750.0, 950.0), face_area=15000.0, frame_width=1920, frame_height=1080)
    assert is_reaction is True


def test_classify_reaction_layout_false_for_large_centered_face():
    # An IRL/selfie-style clip -- face fills a big chunk of a roughly
    # centered frame.
    is_reaction = classify_reaction_layout((960.0, 540.0), face_area=400000.0, frame_width=1920, frame_height=1080)
    assert is_reaction is False


def test_classify_reaction_layout_false_for_small_but_centered_face():
    # Small doesn't automatically mean "webcam box" -- a small face near
    # dead-center (e.g. someone standing far from an IRL camera) shouldn't
    # trigger the split layout either.
    is_reaction = classify_reaction_layout((960.0, 540.0), face_area=15000.0, frame_width=1920, frame_height=1080)
    assert is_reaction is False


def test_classify_reaction_layout_false_for_large_cornered_face():
    # Large but in a corner -- not the small-webcam-box pattern either.
    is_reaction = classify_reaction_layout((1750.0, 950.0), face_area=400000.0, frame_width=1920, frame_height=1080)
    assert is_reaction is False


def test_classify_reaction_layout_handles_zero_area_frame():
    assert classify_reaction_layout((0.0, 0.0), face_area=0.0, frame_width=0, frame_height=0) is False


# ---- classify_reaction_layout: regressions from the 2026-09-11 threshold fix ----
#
# Every case below returned True (i.e. wrongly split an IRL/talking-head clip
# into a reaction layout) under the old 0.18 area / 0.4 corner thresholds.
# Face areas are given as a realistic square face box on a 1920x1080 frame.

def test_classify_reaction_layout_false_for_rule_of_thirds_irl_framing():
    # A person composed on the left third -- ordinary framing, not a webcam
    # overlay. 350x350px face = 5.9% of frame, which the old 18% cap
    # happily called "small", and x=640/y=380 fell inside the old 40%
    # corner margin on both axes.
    assert classify_reaction_layout(
        (640.0, 380.0), face_area=350.0 * 350.0, frame_width=1920, frame_height=1080
    ) is False


def test_classify_reaction_layout_false_for_rule_of_thirds_right():
    assert classify_reaction_layout(
        (1280.0, 380.0), face_area=350.0 * 350.0, frame_width=1920, frame_height=1080
    ) is False


def test_classify_reaction_layout_false_for_seated_off_center_subject():
    # "Just chatting" framing: subject seated right of center and low.
    assert classify_reaction_layout(
        (1250.0, 700.0), face_area=300.0 * 300.0, frame_width=1920, frame_height=1080
    ) is False


def test_classify_reaction_layout_false_for_distant_subject_low_left():
    # Small face (1.1% of frame) because the person is far from an IRL
    # camera, NOT because it is a webcam box -- the size test alone can't
    # tell these apart, which is why the corner test has to be strict.
    assert classify_reaction_layout(
        (500.0, 800.0), face_area=150.0 * 150.0, frame_width=1920, frame_height=1080
    ) is False


def test_classify_reaction_layout_still_true_for_a_real_webcam_box():
    # The case the split layout actually exists for: a genuinely small face
    # pinned in a corner. Must keep working after the tightening above.
    assert classify_reaction_layout(
        (1650.0, 850.0), face_area=200.0 * 200.0, frame_width=1920, frame_height=1080
    ) is True


def test_classify_reaction_layout_still_true_for_top_left_webcam_box():
    assert classify_reaction_layout(
        (250.0, 200.0), face_area=190.0 * 190.0, frame_width=1920, frame_height=1080
    ) is True


# ---- compute_face_zoom_crop ----


def test_compute_face_zoom_crop_matches_target_ratio():
    crop_w, crop_h, x, y = compute_face_zoom_crop(1920, 1080, (1750.0, 950.0), face_scale=100.0, target_ratio=1.125)
    assert crop_w / crop_h == pytest.approx(1.125, abs=0.05)
    assert 0 <= x <= 1920 - crop_w
    assert 0 <= y <= 1080 - crop_h


def test_compute_face_zoom_crop_clamps_to_frame_bounds():
    # A huge padding/face_scale would want a crop bigger than the source --
    # must clamp down to fit, not request an impossible crop.
    crop_w, crop_h, x, y = compute_face_zoom_crop(640, 360, (100.0, 100.0), face_scale=500.0, target_ratio=1.125)
    assert crop_w <= 640
    assert crop_h <= 360


# ---- filtergraph builders ----


def test_build_single_crop_filtergraph_shape():
    graph = build_single_crop_filtergraph(606, 1080, 219, 0, 1080, 1920)
    assert graph == "[0:v]crop=606:1080:219:0,scale=1080:1920,setsar=1[vout]"


def test_build_split_reaction_filtergraph_shape():
    graph = build_split_reaction_filtergraph((300, 267, 10, 20), (1215, 1080, 0, 0), 1080, 960)
    assert graph == (
        "[0:v]crop=300:267:10:20,scale=1080:960,setsar=1[cam];"
        "[0:v]crop=1215:1080:0:0,scale=1080:960,setsar=1[main];"
        "[cam][main]vstack=inputs=2[vout]"
    )


def test_append_subtitles_stage_relabels_output():
    base = build_single_crop_filtergraph(606, 1080, 219, 0, 1080, 1920)
    graph = append_subtitles_stage(base, "/tmp/captions.srt", "FontSize=10")
    assert graph.count("[vout]") == 1  # only the new, final label -- not the relabeled intermediate
    assert "[vpre]" in graph
    assert graph.endswith("[vout]")
    assert "subtitles='/tmp/captions.srt':force_style='FontSize=10'" in graph


# ---- build_title_srt ----


def test_build_title_srt_spans_whole_clip_duration():
    srt = build_title_srt("He Did NOT See That Coming", 12.5)
    assert srt == "1\n00:00:00,000 --> 00:00:12,500\nHe Did NOT See That Coming\n"


def test_build_title_srt_blank_title_returns_empty():
    assert build_title_srt("   ", 10.0) == ""
    assert build_title_srt("", 10.0) == ""


def test_build_title_srt_strips_surrounding_whitespace():
    srt = build_title_srt("  Wow  ", 3.0)
    assert "Wow" in srt
    assert "  Wow  " not in srt


def test_build_title_srt_floors_duration_at_a_hundredth_of_a_second():
    # A zero/negative duration shouldn't produce an inverted or zero-length
    # cue -- clamped to a tiny positive floor, same defensive floor pattern
    # as build_srt's own timestamp handling.
    srt = build_title_srt("Hook", 0.0)
    assert "00:00:00,000 --> 00:00:00,010" in srt


SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "This is insane, wow!"},
    {"start": 2.5, "end": 4.0, "text": "Can you believe that?"},
    {"start": 10.0, "end": 12.0, "text": "outside the window"},
]


def test_build_srt_empty_when_no_overlap():
    assert build_srt(SEGMENTS, 20.0, 30.0) == ""


def test_build_srt_shifts_timestamps_relative_to_clip_start():
    srt = build_srt(SEGMENTS, 2.5, 4.0)
    assert "1\n00:00:00,000 --> 00:00:01,500\nCan you believe that?" in srt


def test_build_srt_clips_segment_to_window_boundary():
    # Segment [0,2] overlapping a window starting at 1.0 -- entry should
    # start at 0 (clipped), not go negative.
    srt = build_srt(SEGMENTS, 1.0, 4.0)
    assert "00:00:00,000 -->" in srt


def test_build_srt_skips_empty_text():
    segments = [{"start": 0.0, "end": 1.0, "text": "   "}]
    assert build_srt(segments, 0.0, 1.0) == ""


def test_build_srt_multiple_entries_numbered_sequentially():
    srt = build_srt(SEGMENTS, 0.0, 5.0)
    assert srt.startswith("1\n")
    assert "\n2\n" in srt


def test_ffmpeg_subtitles_filter_path_escapes_windows_drive_colon():
    escaped = ffmpeg_subtitles_filter_path(r"C:\Users\jeanc\AppData\Local\Temp\captions.srt")
    assert escaped == "C\\:/Users/jeanc/AppData/Local/Temp/captions.srt"


def test_ffmpeg_subtitles_filter_path_leaves_plain_unix_path_mostly_unchanged():
    assert ffmpeg_subtitles_filter_path("/tmp/render-abc/captions.srt") == "/tmp/render-abc/captions.srt"


# ---- fit-frame layout (nothing cropped away) ----


def test_build_fit_frame_filtergraph_ends_in_vout_like_every_other_builder():
    graph = build_fit_frame_filtergraph(1080, 1920)
    assert graph.endswith("[vout]")


def test_build_fit_frame_filtergraph_scales_to_width_and_never_crops_the_content():
    graph = build_fit_frame_filtergraph(1080, 1920)
    # The foreground branch must scale to the target WIDTH with a free
    # (aspect-preserving, even) height -- that's what guarantees the whole
    # source frame survives. A crop on the foreground branch would defeat
    # the entire point of this layout.
    assert "scale=1080:-2" in graph
    assert "[fg]scale=1080:-2" in graph
    # The only crop present belongs to the blurred background branch, which
    # has to cover the full canvas.
    assert graph.count("crop=") == 1
    assert "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" in graph


def test_build_fit_frame_filtergraph_centers_content_over_a_blurred_background():
    graph = build_fit_frame_filtergraph(1080, 1920)
    assert "boxblur=" in graph
    assert "overlay=(W-w)/2:(H-h)/2" in graph


# ---- crop bias override ----


def test_compute_crop_offset_bias_left_pins_to_the_left_edge():
    x, _y = compute_crop_offset(1920, 1080, 608, 1080, bias="left")
    assert x == 0


def test_compute_crop_offset_bias_right_pins_to_the_right_edge():
    x, _y = compute_crop_offset(1920, 1080, 608, 1080, bias="right")
    assert x == 1920 - 608


def test_compute_crop_offset_bias_center_matches_the_default_centered_crop():
    biased = compute_crop_offset(1920, 1080, 608, 1080, bias="center")
    default = compute_crop_offset(1920, 1080, 608, 1080)
    assert biased == default


def test_compute_crop_offset_bias_overrides_a_detected_face():
    # An explicit per-job instruction must win over automatic face
    # detection -- same precedence as camera_layout_mode over
    # classify_reaction_layout.
    face_on_the_right = (1700.0, 540.0)
    unbiased_x, _ = compute_crop_offset(1920, 1080, 608, 1080, focal_point=face_on_the_right)
    biased_x, _ = compute_crop_offset(1920, 1080, 608, 1080, focal_point=face_on_the_right, bias="left")
    assert unbiased_x > 0  # the face really would have pulled the crop right
    assert biased_x == 0


def test_compute_crop_offset_unknown_bias_falls_back_to_existing_behavior():
    # A value that never came through API validation shouldn't silently
    # produce a weird crop -- it just isn't a bias.
    face = (1700.0, 540.0)
    assert compute_crop_offset(1920, 1080, 608, 1080, focal_point=face, bias="sideways") == (
        compute_crop_offset(1920, 1080, 608, 1080, focal_point=face)
    )


def test_compute_crop_offset_bias_is_clamped_to_the_frame():
    # Nothing to crop horizontally (crop_w == width) -- a bias can't push
    # the window off the frame.
    x, _y = compute_crop_offset(1080, 1920, 1080, 1920, bias="right")
    assert x == 0


# ---- multi-part (stitched) clips ----


def test_normalize_parts_accepts_a_valid_two_part_clip():
    assert normalize_parts([[10.0, 15.0], [40.0, 48.0]]) == [(10.0, 15.0), (40.0, 48.0)]


def test_normalize_parts_sorts_by_start():
    assert normalize_parts([[40.0, 48.0], [10.0, 15.0]]) == [(10.0, 15.0), (40.0, 48.0)]


def test_normalize_parts_returns_none_for_single_or_empty():
    # One part isn't a stitch -- going through the concat path for it would
    # add work for no benefit.
    assert normalize_parts([[10.0, 15.0]]) is None
    assert normalize_parts([]) is None
    assert normalize_parts(None) is None


def test_normalize_parts_rejects_overlapping_parts():
    # Overlapping parts would play the same footage twice with a visible
    # jump between the copies.
    assert normalize_parts([[10.0, 20.0], [15.0, 25.0]]) is None


def test_normalize_parts_rejects_malformed_entries():
    assert normalize_parts([[10.0], [20.0, 25.0]]) is None
    assert normalize_parts([[10.0, 5.0], [20.0, 25.0]]) is None  # backwards
    assert normalize_parts([["a", "b"], [20.0, 25.0]]) is None
    assert normalize_parts("not a list") is None


def test_parts_duration_sums_parts_not_the_span():
    # The gaps between parts are exactly what gets cut out -- a span-based
    # duration would be wildly wrong.
    parts = [(10.0, 15.0), (100.0, 108.0)]
    assert parts_duration(parts) == 13.0


def test_build_concat_prefix_trims_and_rebases_every_part():
    prefix = build_concat_prefix([(1.0, 3.0), (6.0, 8.5)], include_audio=False)
    assert "trim=start=1.000:end=3.000" in prefix
    assert "trim=start=6.000:end=8.500" in prefix
    # setpts rebasing is what stops concat producing huge gaps matching the
    # removed material.
    assert prefix.count("setpts=PTS-STARTPTS") == 2
    assert "concat=n=2:v=1:a=0[vsrc]" in prefix


def test_build_concat_prefix_omits_the_audio_branch_when_asked():
    prefix = build_concat_prefix([(1.0, 3.0), (6.0, 8.5)], include_audio=False)
    # Referencing [0:a] on a source with no audio track fails the whole
    # ffmpeg command, which is why this is opt-in.
    assert "[0:a]" not in prefix
    assert "[asrc]" not in prefix


def test_build_concat_prefix_includes_a_matching_audio_branch():
    prefix = build_concat_prefix([(1.0, 3.0), (6.0, 8.5)], include_audio=True)
    assert "atrim=start=1.000:end=3.000" in prefix
    assert "asetpts=PTS-STARTPTS" in prefix
    assert "concat=n=2:v=0:a=1[asrc]" in prefix


def test_retarget_source_label_rewrites_every_source_reference():
    # The split layout reads [0:v] twice -- both branches have to be
    # retargeted or half the graph still reads the untrimmed input.
    graph = build_split_reaction_filtergraph((10, 10, 0, 0), (20, 20, 1, 1), 1080, 960)
    assert graph.count("[0:v]") == 2
    retargeted = retarget_source_label(graph, "vsrc")
    assert "[0:v]" not in retargeted
    assert retargeted.count("[vsrc]") == 2


def test_retarget_source_label_works_for_every_layout_builder():
    for graph in (
        build_single_crop_filtergraph(202, 360, 219, 0, 1080, 1920),
        build_fit_frame_filtergraph(1080, 1920),
        build_split_reaction_filtergraph((10, 10, 0, 0), (20, 20, 1, 1), 1080, 960),
    ):
        assert "[0:v]" not in retarget_source_label(graph, "vsrc")


def test_build_multipart_srt_remaps_later_parts_onto_the_stitched_timeline():
    segments = [
        {"start": 10.0, "end": 12.0, "text": "first part line"},
        {"start": 50.0, "end": 52.0, "text": "second part line"},
    ]
    # Part one is 10-13 (3s); part two starts at 50, so its caption must
    # play at 3s into the stitched clip, not at 40s.
    srt = build_multipart_srt(segments, [(10.0, 13.0), (50.0, 53.0)])
    assert "first part line" in srt
    assert "second part line" in srt
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "00:00:03,000 --> 00:00:05,000" in srt


def test_build_multipart_srt_is_empty_when_no_part_has_text():
    segments = [{"start": 100.0, "end": 102.0, "text": "elsewhere entirely"}]
    assert build_multipart_srt(segments, [(10.0, 13.0), (50.0, 53.0)]) == ""


# ---- crop_from_facecam_rect (hand-marked facecam box) ----
#
# The split layout's top panel is 1080x960, so every case below resolves the
# marked box against that ratio.

_HALF_RATIO = 1080 / 960
_TYPICAL_RECT = {"x": 0.72, "y": 0.60, "w": 0.26, "h": 0.36}


def test_crop_from_facecam_rect_contains_the_whole_marked_box():
    # The one property that matters: the creator drew a box around the
    # facecam, so the crop must never cut into it.
    w, h, x, y = crop_from_facecam_rect(1920, 1080, _TYPICAL_RECT, _HALF_RATIO)
    assert x <= _TYPICAL_RECT["x"] * 1920
    assert y <= _TYPICAL_RECT["y"] * 1080
    assert x + w >= (_TYPICAL_RECT["x"] + _TYPICAL_RECT["w"]) * 1920
    assert y + h >= (_TYPICAL_RECT["y"] + _TYPICAL_RECT["h"]) * 1080


def test_crop_from_facecam_rect_matches_the_target_aspect_ratio():
    w, h, _, _ = crop_from_facecam_rect(1920, 1080, _TYPICAL_RECT, _HALF_RATIO)
    assert abs((w / h) - _HALF_RATIO) < 0.02


def test_crop_from_facecam_rect_stays_inside_the_frame():
    for rect in (
        {"x": 0.0, "y": 0.0, "w": 0.15, "h": 0.20},     # hard against the top-left
        {"x": 0.85, "y": 0.80, "w": 0.15, "h": 0.20},   # hard against the bottom-right
        _TYPICAL_RECT,
    ):
        w, h, x, y = crop_from_facecam_rect(1920, 1080, rect, _HALF_RATIO)
        assert x >= 0 and y >= 0
        assert x + w <= 1920 and y + h <= 1080


def test_crop_from_facecam_rect_allows_a_zero_offset():
    # Regression: an earlier version floored offsets to 2 (the minimum for a
    # crop *size*), nudging the window off a facecam pinned to the very edge
    # of the frame -- i.e. the most common case.
    _, _, x, y = crop_from_facecam_rect(
        1920, 1080, {"x": 0.0, "y": 0.0, "w": 0.15, "h": 0.20}, _HALF_RATIO
    )
    assert (x, y) == (0, 0)


def test_crop_from_facecam_rect_returns_even_numbers():
    # yuv420p needs even dimensions; an odd crop makes ffmpeg fail or shift
    # a pixel silently.
    for value in crop_from_facecam_rect(1919, 1079, _TYPICAL_RECT, _HALF_RATIO):
        assert value % 2 == 0


def test_crop_from_facecam_rect_shrinks_a_box_too_large_for_the_frame():
    # A box covering the whole frame can't be grown to the target ratio, so
    # the largest correctly-proportioned window is used instead of returning
    # something off-frame or stretched.
    w, h, x, y = crop_from_facecam_rect(
        1920, 1080, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}, _HALF_RATIO
    )
    assert w <= 1920 and h <= 1080
    assert x + w <= 1920 and y + h <= 1080
    assert abs((w / h) - _HALF_RATIO) < 0.02


def test_crop_from_facecam_rect_grows_a_wide_box_vertically():
    # Box wider than the target ratio: height grows, width is preserved.
    w, h, _, _ = crop_from_facecam_rect(
        1920, 1080, {"x": 0.1, "y": 0.4, "w": 0.6, "h": 0.1}, _HALF_RATIO
    )
    assert h > 0.1 * 1080
    assert abs((w / h) - _HALF_RATIO) < 0.02


def test_crop_from_facecam_rect_is_resolution_independent():
    # Same normalized mark, two source resolutions -> same relative window.
    big = crop_from_facecam_rect(1920, 1080, _TYPICAL_RECT, _HALF_RATIO)
    small = crop_from_facecam_rect(1280, 720, _TYPICAL_RECT, _HALF_RATIO)
    assert abs((big[0] / 1920) - (small[0] / 1280)) < 0.01
    assert abs((big[2] / 1920) - (small[2] / 1280)) < 0.01


@pytest.mark.parametrize(
    "rect",
    [
        {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.2},   # no width
        {"x": 0.0, "y": 0.0},                       # missing keys
        {"x": "a", "y": 0.0, "w": 0.1, "h": 0.1},   # not a number
    ],
)
def test_crop_from_facecam_rect_rejects_a_malformed_rect(rect):
    # Callers treat ValueError as "behave as if nothing was marked" rather
    # than failing the render -- see app/workers/rendering.py.
    with pytest.raises(ValueError):
        crop_from_facecam_rect(1920, 1080, rect, _HALF_RATIO)
