"""Real ffmpeg-driven tests for app.core.stt.chunking -- no network, no STT
model needed, so these run in any environment (unlike the local-Whisper
tests in test_transcription_worker.py, which need a model download).

Builds an 8s synthetic WAV: 2s tone, 1s silence, 2s tone, 1s silence, 2s
tone -- so silence sits at [2,3) and [5,6), with midpoints 2.5 and 5.5.
"""
import os
import shutil
import subprocess

import pytest

from app.core.stt.chunking import plan_chunk_boundaries, split_audio


@pytest.fixture
def tone_silence_wav(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")
    path = tmp_path / "tone_silence.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-filter_complex", "[0][1][2][3][4]concat=n=5:v=0:a=1[out]",
            "-map", "[out]", "-ac", "1", "-ar", "16000",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return str(path)


def test_no_boundaries_when_shorter_than_target(tone_silence_wav):
    assert plan_chunk_boundaries(tone_silence_wav, duration_seconds=8.0, target_chunk_seconds=30.0) == []


def test_boundaries_snap_to_silence_midpoints(tone_silence_wav):
    boundaries = plan_chunk_boundaries(
        tone_silence_wav, duration_seconds=8.0, target_chunk_seconds=3.0, search_window_seconds=2.0
    )
    assert len(boundaries) == 2
    assert boundaries[0] == pytest.approx(2.5, abs=0.1)
    assert boundaries[1] == pytest.approx(5.5, abs=0.1)


def test_falls_back_to_hard_cut_when_no_silence_nearby(tone_silence_wav):
    # A tiny search window won't reach either silence gap from these targets,
    # so every cut should fall back to the raw target rather than finding
    # (or wrongly snapping to) a distant silence.
    boundaries = plan_chunk_boundaries(
        tone_silence_wav, duration_seconds=8.0, target_chunk_seconds=3.0, search_window_seconds=0.05
    )
    assert boundaries == pytest.approx([3.0, 6.0], abs=0.05)


def test_split_audio_produces_contiguous_chunks_covering_full_duration(tone_silence_wav, tmp_path):
    boundaries = plan_chunk_boundaries(
        tone_silence_wav, duration_seconds=8.0, target_chunk_seconds=3.0, search_window_seconds=2.0
    )
    out_dir = tmp_path / "chunks"
    out_dir.mkdir()
    chunks = split_audio(tone_silence_wav, boundaries, str(out_dir))

    assert len(chunks) == len(boundaries) + 1
    assert chunks[0].offset_seconds == 0.0
    assert [c.offset_seconds for c in chunks[1:]] == boundaries
    for chunk in chunks:
        assert os.path.exists(chunk.path)
