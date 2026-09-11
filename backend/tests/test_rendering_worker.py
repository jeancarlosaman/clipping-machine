"""Exercises app.workers.rendering.run() directly -- real DB, real storage,
real ffmpeg (crop/scale/caption burn-in), same spirit as
test_segmentation_worker.py / test_scoring_worker.py. Only the transcript
is fabricated.
"""
import json
import subprocess
import uuid

import pytest

from app.core.config import settings
from app.core.storage import storage
from app.db.models import CandidateSegment, RenderedClip, StreamJob, Transcript
from app.workers import rendering

TRANSCRIPT_SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "hello there"},
    {"start": 1.5, "end": 3.0, "text": "this is a test clip"},
]


@pytest.fixture(autouse=True)
def _disable_llm_captions_for_rendering_tests(monkeypatch):
    # rendering.run() now calls generate_caption_annotation synchronously
    # as part of the render itself (see that worker's module docstring) --
    # force the deterministic heuristic path here so these tests never make
    # a real network call, regardless of what OPENAI_API_KEY happens to be
    # set to in whatever environment runs them. The LLM path itself is
    # already covered by test_caption_generation.py's mocked-client tests.
    monkeypatch.setattr(settings, "enable_llm_captions", False)


@pytest.fixture
def landscape_video_with_audio(tmp_path):
    """A short real 16:9 video with a real audio track, for exercising the
    full crop/scale/caption/encode pipeline end to end."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    path = tmp_path / "landscape.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", "-shortest",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def _make_selected_candidate(db_session, user, video_path, transcript_segments, *, start=0.0, end=3.0):
    object_key = "raw/rendering-test.mp4"
    storage.put_file(str(video_path), object_key)

    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key=object_key, status="scored")
    db_session.add(job)
    db_session.flush()

    if transcript_segments is not None:
        db_session.add(
            Transcript(stream_job_id=job.id, provider="fake", full_text="test", segments=transcript_segments)
        )

    candidate = CandidateSegment(
        stream_job_id=job.id, start_seconds=start, end_seconds=end, status="selected"
    )
    db_session.add(candidate)
    db_session.flush()

    clip = RenderedClip(candidate_segment_id=candidate.id, stream_job_id=job.id, status="pending")
    db_session.add(clip)
    db_session.commit()
    db_session.refresh(candidate)
    db_session.refresh(clip)
    return job, candidate, clip


def test_rendering_success_produces_video_and_thumbnail(db_session, user, landscape_video_with_audio):
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.object_key
    assert clip.thumbnail_key
    assert clip.duration_seconds == pytest.approx(3.0)
    assert clip.flag_reasons == []
    assert clip.retry_count == 1
    assert storage.exists(clip.object_key)
    assert storage.exists(clip.thumbnail_key)

    # Rendered output should actually be a real, playable 9:16 video --
    # verify with ffprobe rather than just trusting the ffmpeg exit code.
    local_path = storage.get_local_path(clip.object_key, download_to="")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", local_path,
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["width"] == 1080
    assert stream["height"] == 1920


def test_rendering_thumbnail_falls_back_to_fixed_frame_when_selection_finds_nothing(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # app.core.thumbnail_selection.pick_best_frame returning None (every
    # sampled candidate was unreadable) must not fail the render -- it
    # should fall back to the original fixed-frame ffmpeg extract, same as
    # before thumbnail selection existed.
    monkeypatch.setattr(rendering, "pick_best_frame", lambda paths: None)
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.thumbnail_key
    assert storage.exists(clip.thumbnail_key)


def test_rendering_thumbnail_falls_back_to_fixed_frame_when_selection_raises(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # A hard failure inside thumbnail selection (cv2 blowing up, etc.) is a
    # reviewer-experience issue, not a reason to lose an otherwise-good
    # clip -- same posture as the pre-existing "thumbnail extraction failed
    # entirely" handling around _extract_thumbnail's caller.
    def raising_pick_best_frame(paths):
        raise RuntimeError("cv2 exploded")

    monkeypatch.setattr(rendering, "pick_best_frame", raising_pick_best_frame)
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.thumbnail_key
    assert storage.exists(clip.thumbnail_key)


def test_rendering_generates_annotation_synchronously_without_enqueueing(
    db_session, user, landscape_video_with_audio
):
    # Title/hashtags/caption/explanation generation moved from an
    # after-render async enqueue to a synchronous call inside rendering.run
    # itself, specifically so the title can be burned into the clip (see
    # the module docstring) -- rendering.py no longer imports enqueue or
    # app.workers.caption_generation at all, so there's nothing left to
    # monkeypatch/assert-was-called; instead this asserts the *outcome*
    # (annotation already present) is there by the time status='rendered'.
    assert not hasattr(rendering, "enqueue")
    assert not hasattr(rendering, "caption_generation")

    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    db_session.refresh(candidate)
    assert clip.status == "rendered"
    assert candidate.llm_annotation is not None
    assert candidate.llm_annotation["source"] == "heuristic_fallback"  # LLM disabled by the autouse fixture
    assert candidate.llm_annotation["title"]  # heuristic_title never returns empty
    assert candidate.llm_annotation["hashtags"]
    assert clip.caption_text == candidate.llm_annotation["caption"]


def test_rendering_burns_in_title_banner_by_default(db_session, user, landscape_video_with_audio, monkeypatch):
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )

    from app.workers.common import run_subprocess as real_run_subprocess

    captured_graphs = []

    def spy(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured_graphs.append(cmd[cmd.index("-filter_complex") + 1])
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr(rendering, "run_subprocess", spy)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.flag_reasons == []  # both burn-in stages actually succeeded, nothing degraded
    graph = captured_graphs[0]
    # Two chained subtitles stages: transcript captions (Alignment=2, see
    # _CAPTION_STYLE) then the title banner (Alignment=6, see _TITLE_STYLE --
    # legacy-SSA-numbered top-center, not ASS numpad's 8, see that
    # constant's comment for why).
    assert graph.count("subtitles=") == 2
    assert "Alignment=2" in graph
    assert "Alignment=6" in graph


def test_rendering_skips_title_burn_in_when_overlay_disabled(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    monkeypatch.setattr(settings, "enable_clip_title_overlay", False)

    from app.workers.common import run_subprocess as real_run_subprocess

    captured_graphs = []

    def spy(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured_graphs.append(cmd[cmd.index("-filter_complex") + 1])
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr(rendering, "run_subprocess", spy)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    db_session.refresh(candidate)
    assert clip.status == "rendered"
    graph = captured_graphs[0]
    assert graph.count("subtitles=") == 1  # captions only -- no title stage burned in
    assert "Alignment=6" not in graph
    # The toggle only controls the burned-in pixels -- the title text is
    # still generated and stored either way.
    assert candidate.llm_annotation["title"]


def test_rendering_skips_face_detection_when_disabled(db_session, user, landscape_video_with_audio, monkeypatch):
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    monkeypatch.setattr(settings, "enable_face_aware_crop", False)

    def _boom(*args, **kwargs):
        raise AssertionError("face detection should not run when disabled")

    monkeypatch.setattr(rendering, "estimate_face_profile", _boom)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"


def test_rendering_survives_face_detection_failure(db_session, user, landscape_video_with_audio, monkeypatch):
    # Face-aware cropping is best-effort -- any failure in it must degrade
    # to the plain centered crop, never break the render itself.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    monkeypatch.setattr(
        rendering,
        "_extract_face_sample_frames",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg exploded")),
    )

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"


def test_rendering_offsets_crop_toward_detected_focal_point(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # Force a focal point far off-center and verify the actual ffmpeg crop
    # filter used a non-centered x offset -- proves the wiring (not just
    # the pure compute_crop_offset math already covered by
    # test_rendering_logic.py) actually takes effect end to end.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    # area=60000 is comfortably above the 6% cap for the 640x360 fixture
    # frame (230400 * 0.06 = 13824) so classify_reaction_layout stays False
    # regardless of the (deliberately off-center) face position -- this
    # test is specifically about the single-crop offset math, the split
    # layout has its own coverage.
    monkeypatch.setattr(
        rendering,
        "estimate_face_profile",
        lambda paths: {"center": (50.0, 180.0), "area": 60000.0, "frame_size": (640, 360)},
    )

    from app.workers.common import run_subprocess as real_run_subprocess

    captured_graphs = []

    def spy(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured_graphs.append(cmd[cmd.index("-filter_complex") + 1])
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr(rendering, "run_subprocess", spy)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert captured_graphs, "expected at least one ffmpeg call with -filter_complex"
    # 640x360 source -> crop_w=202 (see compute_vertical_crop's own tests);
    # centered would be (640-202)//2 = 219. A focal point at x=50 should
    # pull the crop's x offset down toward the left edge instead.
    graph = captured_graphs[0]
    crop_segment = graph.split(",")[0]  # "[0:v]crop=W:H:X:Y" -- strip the
    # "[0:v]" input-label prefix before splitting on ":", or its own colon
    # throws off the field indices.
    crop_values = crop_segment.split("crop=")[1]
    x_offset = int(crop_values.split(":")[2])
    assert x_offset < 219
    assert x_offset == 0  # clamped: 50 - 202/2 would be negative


def test_rendering_uses_split_layout_for_small_cornered_face(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # Force a small, corner-positioned face profile (the classic reaction-
    # webcam-box shape) and verify the render actually takes the split
    # vstack path end to end -- not just that classify_reaction_layout
    # returns True in isolation (already covered by test_rendering_logic.py).
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    # 640x360 fixture frame -> area must be <= 18% of 230400 = 41472, and
    # the center must sit within 40% of the frame size from some corner on
    # both axes. Bottom-right corner, well within both thresholds.
    monkeypatch.setattr(
        rendering,
        "estimate_face_profile",
        lambda paths: {"center": (580.0, 320.0), "area": 8000.0, "frame_size": (640, 360)},
    )

    from app.workers.common import run_subprocess as real_run_subprocess

    captured_graphs = []

    def spy(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured_graphs.append(cmd[cmd.index("-filter_complex") + 1])
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr(rendering, "run_subprocess", spy)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.object_key
    assert storage.exists(clip.object_key)

    graph = captured_graphs[0]
    assert "[cam]" in graph and "[main]" in graph
    assert "vstack=inputs=2" in graph

    local_path = storage.get_local_path(clip.object_key, download_to="")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", local_path,
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["width"] == 1080
    assert stream["height"] == 1920


def test_rendering_camera_layout_mode_single_crop_overrides_reaction_classification(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # Real user-reported issue: the auto classifier kept detecting a
    # "reaction cam" on VODs that don't actually have one, splitting clips
    # and needlessly shrinking how much of the actual content is shown.
    # camera_layout_mode="single_crop" should force single-crop even with
    # the exact same small-cornered face profile that
    # test_rendering_uses_split_layout_for_small_cornered_face proves DOES
    # trigger the split layout under "auto".
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    job.camera_layout_mode = "single_crop"
    db_session.commit()

    monkeypatch.setattr(
        rendering,
        "estimate_face_profile",
        lambda paths: {"center": (580.0, 320.0), "area": 8000.0, "frame_size": (640, 360)},
    )

    from app.workers.common import run_subprocess as real_run_subprocess

    captured_graphs = []

    def spy(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured_graphs.append(cmd[cmd.index("-filter_complex") + 1])
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr(rendering, "run_subprocess", spy)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    graph = captured_graphs[0]
    assert "vstack" not in graph
    assert "[cam]" not in graph


def test_rendering_camera_layout_mode_split_reaction_overrides_classifier_miss(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # Mirror case: a face big/centered enough that "auto" would NOT
    # classify it as a reaction cam (same profile
    # test_rendering_offsets_crop_toward_detected_focal_point uses to stay
    # single-crop) -- camera_layout_mode="split_reaction" should force the
    # split anyway, for a creator who knows this VOD has a facecam and
    # whose specific webcam size/position the classifier's thresholds
    # guess wrong for.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    job.camera_layout_mode = "split_reaction"
    db_session.commit()

    monkeypatch.setattr(
        rendering,
        "estimate_face_profile",
        lambda paths: {"center": (50.0, 180.0), "area": 60000.0, "frame_size": (640, 360)},
    )

    from app.workers.common import run_subprocess as real_run_subprocess

    captured_graphs = []

    def spy(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured_graphs.append(cmd[cmd.index("-filter_complex") + 1])
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr(rendering, "run_subprocess", spy)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    graph = captured_graphs[0]
    assert "[cam]" in graph and "[main]" in graph
    assert "vstack=inputs=2" in graph


def test_rendering_camera_layout_mode_split_reaction_falls_back_with_no_face_found(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # Can't split without a face to zoom the "cam" half on -- split_reaction
    # degrades to single_crop on a clip with no detected face, same as
    # "auto" already does, rather than erroring or fabricating a crop.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    job.camera_layout_mode = "split_reaction"
    db_session.commit()

    monkeypatch.setattr(rendering, "estimate_face_profile", lambda paths: None)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"


def test_rendering_camera_layout_mode_split_reaction_respects_global_kill_switch(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # settings.enable_reaction_split_layout=False is a hard kill switch --
    # a per-job "please split" override shouldn't silently defeat an
    # operator's decision to turn the whole feature off.
    monkeypatch.setattr(settings, "enable_reaction_split_layout", False)
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    job.camera_layout_mode = "split_reaction"
    db_session.commit()

    monkeypatch.setattr(
        rendering,
        "estimate_face_profile",
        lambda paths: {"center": (580.0, 320.0), "area": 8000.0, "frame_size": (640, 360)},
    )

    from app.workers.common import run_subprocess as real_run_subprocess

    captured_graphs = []

    def spy(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured_graphs.append(cmd[cmd.index("-filter_complex") + 1])
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr(rendering, "run_subprocess", spy)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    graph = captured_graphs[0]
    assert "vstack" not in graph


def test_rendering_without_transcript_still_produces_captionless_video(db_session, user, landscape_video_with_audio):
    # No Transcript row at all -- build_srt has nothing to work with, so
    # this should render fine, just without burned-in captions.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, None
    )

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.object_key
    assert "captions_failed" not in clip.flag_reasons


def test_rendering_missing_candidate_is_a_noop(db_session):
    rendering.run(str(uuid.uuid4()))


def test_rendering_missing_clip_row_is_a_noop(db_session, user, landscape_video_with_audio):
    object_key = "raw/no-clip-row.mp4"
    storage.put_file(str(landscape_video_with_audio), object_key)
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key=object_key, status="scored")
    db_session.add(job)
    db_session.flush()
    candidate = CandidateSegment(stream_job_id=job.id, start_seconds=0, end_seconds=2, status="selected")
    db_session.add(candidate)
    db_session.commit()
    db_session.refresh(candidate)

    # No RenderedClip row created -- scoring always creates one before
    # enqueuing, so this simulates the row having been removed/never made.
    rendering.run(str(candidate.id))  # should not raise


def test_rendering_permanent_failure_on_unreadable_source(db_session, user):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/does-not-exist.mp4", status="scored")
    db_session.add(job)
    db_session.flush()
    candidate = CandidateSegment(stream_job_id=job.id, start_seconds=0, end_seconds=2, status="selected")
    db_session.add(candidate)
    db_session.flush()
    clip = RenderedClip(candidate_segment_id=candidate.id, stream_job_id=job.id, status="pending")
    db_session.add(clip)
    db_session.commit()
    db_session.refresh(clip)

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "failed"
    assert clip.last_error


def test_on_failure_marks_clip_failed(db_session, user):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="scored")
    db_session.add(job)
    db_session.flush()
    candidate = CandidateSegment(stream_job_id=job.id, start_seconds=0, end_seconds=2, status="selected")
    db_session.add(candidate)
    db_session.flush()
    clip = RenderedClip(candidate_segment_id=candidate.id, stream_job_id=job.id, status="rendering")
    db_session.add(clip)
    db_session.commit()
    db_session.refresh(clip)

    fake_rq_job = type("FakeRqJob", (), {"args": [str(candidate.id)]})()
    rendering.on_failure(fake_rq_job, None, RuntimeError, RuntimeError("exhausted"), None)

    db_session.refresh(clip)
    assert clip.status == "failed"
    assert "exhausted" in clip.last_error


def test_rendering_fit_frame_layout_renders_without_cropping(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # The whole point of fit_frame: a real end-to-end render that keeps the
    # full 16:9 source, still producing a valid 1080x1920 output.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    job.camera_layout_mode = "fit_frame"
    db_session.commit()

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    local_path = storage.get_local_path(clip.object_key, download_to="")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", local_path,
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["width"] == 1080
    assert stream["height"] == 1920


def test_rendering_fit_frame_ignores_face_detection_entirely(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    # Nothing is cropped in this layout, so there is no crop to center on a
    # face -- a detected reaction-shaped face must not flip it back to the
    # split layout.
    monkeypatch.setattr(settings, "enable_face_aware_crop", True)
    monkeypatch.setattr(settings, "enable_reaction_split_layout", True)
    monkeypatch.setattr(
        "app.workers.rendering.estimate_face_profile",
        lambda paths: {"center": (60.0, 40.0), "area": 2500.0, "frame_size": (640, 360)},
    )
    graphs = []
    real_render = rendering._render
    monkeypatch.setattr(
        rendering, "_render",
        lambda *a, **k: (graphs.append(a[4]), real_render(*a, **k))[1],
    )

    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    job.camera_layout_mode = "fit_frame"
    db_session.commit()

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    # fit_frame's signature: an overlay over a blurred background, and no
    # vstack (which is what the split layout would have produced).
    assert "overlay=" in graphs[0]
    assert "boxblur=" in graphs[0]
    assert "vstack" not in graphs[0]


def test_rendering_crop_bias_shifts_the_single_crop_window(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    graphs = []
    real_render = rendering._render
    monkeypatch.setattr(
        rendering, "_render",
        lambda *a, **k: (graphs.append(a[4]), real_render(*a, **k))[1],
    )

    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS
    )
    job.crop_bias = "left"
    db_session.commit()

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    # 640x360 source -> 9:16 crop is 202 wide; "left" pins x to 0, where the
    # unbiased centered crop would have started well inside the frame.
    assert ":0:0," in graphs[0] or graphs[0].startswith("[0:v]crop=202:360:0:0")


def test_rendering_stitched_multipart_clip_produces_the_summed_duration(
    db_session, user, landscape_video_with_audio
):
    # Real end-to-end stitched render: two non-adjacent parts of a 4s
    # source joined into one clip. The output must play for the SUM of the
    # parts (1.0 + 1.5 = 2.5s), not the 3.5s span they're drawn from.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS, start=0.0, end=3.5
    )
    candidate.parts = [[0.0, 1.0], [2.0, 3.5]]
    db_session.commit()

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.duration_seconds == pytest.approx(2.5)

    local_path = storage.get_local_path(clip.object_key, download_to="")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-show_entries", "stream=width,height,codec_type", "-of", "json", local_path,
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(probe.stdout)
    assert float(data["format"]["duration"]) == pytest.approx(2.5, abs=0.35)
    video = [s for s in data["streams"] if s.get("codec_type") == "video"][0]
    assert (video["width"], video["height"]) == (1080, 1920)
    # The source has audio, so the stitched clip must too -- the concat
    # audio branch is only built when the source actually has a track.
    assert any(s.get("codec_type") == "audio" for s in data["streams"])


def test_rendering_stitched_clip_uses_the_concat_prefix_and_retargets_the_layout(
    db_session, user, landscape_video_with_audio, monkeypatch
):
    commands = []
    from app.workers.common import run_subprocess as real_run_subprocess

    def spy(cmd, **kwargs):
        commands.append(cmd)
        return real_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr("app.workers.rendering.run_subprocess", spy)

    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS, start=0.0, end=3.5
    )
    candidate.parts = [[0.0, 1.0], [2.0, 3.5]]
    db_session.commit()

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    render_cmd = next(c for c in commands if "-filter_complex" in c and "[vout]" in c)
    graph = render_cmd[render_cmd.index("-filter_complex") + 1]
    assert "trim=start=0.000:end=1.000" in graph
    assert "trim=start=2.000:end=3.500" in graph
    assert "concat=n=2" in graph
    # The layout must read from the concatenated stream, not the raw input.
    assert "[0:v]crop" not in graph
    assert "[vsrc]" in graph
    # No outer -ss/-t: those would shift the timestamps the trims are
    # expressed in and silently cut the wrong material.
    assert "-ss" not in render_cmd
    assert "-t" not in render_cmd


def test_rendering_stitched_clip_survives_a_source_with_no_audio(
    db_session, user, tmp_path
):
    # A filtergraph referencing [0:a] fails outright on a source with no
    # audio track, so the concat prefix must omit the audio branch there.
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent),
        ],
        capture_output=True, check=True,
    )

    job, candidate, clip = _make_selected_candidate(
        db_session, user, silent, TRANSCRIPT_SEGMENTS, start=0.0, end=3.5
    )
    candidate.parts = [[0.0, 1.0], [2.0, 3.5]]
    db_session.commit()

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.duration_seconds == pytest.approx(2.5)


def test_rendering_ignores_malformed_parts_and_renders_the_plain_window(
    db_session, user, landscape_video_with_audio
):
    # A bad parts value degrades the clip to its ordinary [start, end]
    # window rather than failing the render.
    job, candidate, clip = _make_selected_candidate(
        db_session, user, landscape_video_with_audio, TRANSCRIPT_SEGMENTS, start=0.0, end=3.0
    )
    candidate.parts = [[1.0, 2.0]]  # only one part -- not a stitch
    db_session.commit()

    rendering.run(str(candidate.id))

    db_session.refresh(clip)
    assert clip.status == "rendered"
    assert clip.duration_seconds == pytest.approx(3.0)
