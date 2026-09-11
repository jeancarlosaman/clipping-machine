"""Audio chunking for STT providers with a request-size limit.

Only app/core/stt/openai_api.py needs this -- the OpenAI transcription
endpoint enforces a hard request size limit (~25MB), which most VOD-length
16kHz mono WAVs exceed. faster-whisper (app/core/stt/local_whisper.py) reads
the whole file itself and doesn't need any of this.

Splits near `target_chunk_seconds` boundaries, snapped to the nearest
detected silence (via ffmpeg's silencedetect filter) so a cut doesn't land
mid-word. Falls back to a hard cut at the target boundary if no silence is
found within `search_window_seconds` of it -- an occasional mid-word cut is
better than a chunk that violates the API's size limit.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

from app.workers.common import logger, run_subprocess

_SILENCE_RE = re.compile(r"silence_(start|end): (?P<ts>-?[0-9.]+)")


@dataclass
class AudioChunk:
    path: str
    offset_seconds: float


def _detect_silences(wav_path: str, noise_db: str = "-30dB", min_duration: float = 0.3) -> list[tuple[float, float]]:
    """Return (silence_start, silence_end) windows via ffmpeg's silencedetect filter.

    ffmpeg with `-f null -` "fails" in the sense of producing no output file
    even on success -- the silencedetect markers are written to stderr
    regardless of exit code, so exit code isn't a useful success signal here,
    only the stderr content is.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-i", wav_path,
            "-af", f"silencedetect=noise={noise_db}:d={min_duration}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    starts: list[float] = []
    ends: list[float] = []
    for line in result.stderr.splitlines():
        match = _SILENCE_RE.search(line)
        if not match:
            continue
        ts = float(match.group("ts"))
        if "silence_start" in line:
            starts.append(ts)
        else:
            ends.append(ts)
    # ffmpeg always closes a silence_start with a silence_end, except when
    # the file ends while still silent -- drop a trailing unmatched start
    # rather than mis-pairing it with the next file's first end.
    pair_count = min(len(starts), len(ends))
    return list(zip(starts[:pair_count], ends[:pair_count]))


def _nearest_silence_midpoint(
    silences: list[tuple[float, float]], target: float, search_window: float
) -> float | None:
    candidates = [
        (start + end) / 2
        for start, end in silences
        if abs(((start + end) / 2) - target) <= search_window
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda mid: abs(mid - target))


def plan_chunk_boundaries(
    wav_path: str,
    duration_seconds: float,
    target_chunk_seconds: float = 600.0,
    search_window_seconds: float = 30.0,
) -> list[float]:
    """Interior cut points (seconds) splitting the file into ~target_chunk_seconds pieces.

    Callers slice [0, b0), [b0, b1), ..., [bN, duration) -- this returns only
    the interior boundaries, not the implicit 0 / duration endpoints.
    """
    if duration_seconds <= target_chunk_seconds:
        return []

    silences = _detect_silences(wav_path)
    boundaries: list[float] = []
    target = target_chunk_seconds
    while target < duration_seconds:
        cut = _nearest_silence_midpoint(silences, target, search_window_seconds)
        if cut is None:
            logger.warning("stt.chunking.no_silence_near_target", target=target)
            cut = target
        # Guard against a degenerate/duplicate cut if silence search pushed
        # us backwards past the previous boundary.
        if not boundaries or cut > boundaries[-1] + 1.0:
            boundaries.append(cut)
        target = cut + target_chunk_seconds
    return boundaries


def split_audio(wav_path: str, boundaries: list[float], out_dir: str) -> list[AudioChunk]:
    """Cut `wav_path` at `boundaries` into separate WAV files under `out_dir`.

    Uses accurate (post-input) `-ss`/`-to` seeking rather than fast
    (pre-input) seeking, since correctness matters more than speed here and
    VOD audio chunk counts are small (a few per hour of stream) -- not
    optimized for files that would need hundreds of chunks.
    """
    cut_points: list[float | None] = [0.0, *boundaries, None]
    chunks = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        out_path = os.path.join(out_dir, f"chunk_{i:03d}.wav")
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        cmd += ["-ac", "1", "-ar", "16000", out_path]
        run_subprocess(cmd)
        chunks.append(AudioChunk(path=out_path, offset_seconds=start))
    return chunks
