"""Ingest worker -- architecture doc §7.

Trigger:  stream_job.created (enqueued by POST /api/v1/stream-jobs)
Input:    stream_job_id
Output:   extracted audio track in object storage, probed duration_seconds
State:    queued -> ingesting -> ingested | failed_ingest
Retries:  3x exponential backoff on transient errors (see app.core.queue.enqueue);
          codec/container errors are treated as permanent (see run()).

This is the one worker implemented end-to-end in the first pass -- see
architecture doc §11 "next action". The rest of the pipeline stages
(app/workers/transcription.py etc.) are stubs with the same shape.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid

from app.core.queue import enqueue
from app.core.storage import storage
from app.db.models import StreamJob
from app.workers import transcription
from app.workers.common import audio_object_key, db_session, estimate_job_timeout_seconds, logger, run_subprocess


class PermanentIngestError(Exception):
    """Raised for errors that retrying will never fix (e.g. corrupt/unsupported file).

    Distinct from a bare RuntimeError so `run()` can decide not to let RQ
    burn retries on something that will fail identically every time.
    """


def _probe_duration_seconds(local_path: str) -> float:
    # A non-zero ffprobe exit here overwhelmingly means the file is
    # corrupt/unsupported, not a transient failure -- treat it as permanent
    # so a bad upload doesn't burn 3 pointless RQ retries before failing.
    try:
        result = run_subprocess(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                local_path,
            ]
        )
    except RuntimeError as exc:
        raise PermanentIngestError(f"ffprobe could not read the file: {exc}") from exc

    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise PermanentIngestError(f"Could not read duration from ffprobe output: {exc}") from exc


def _extract_audio(local_video_path: str, local_audio_path: str) -> None:
    # Mono 16kHz WAV: small, and the format OpenAI STT and most ASR expect --
    # decided here once so the transcription worker doesn't need to care.
    run_subprocess(
        [
            "ffmpeg", "-y",
            "-i", local_video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            local_audio_path,
        ]
    )


def run(stream_job_id: str) -> None:
    log = logger.bind(stream_job_id=stream_job_id, worker="ingest")
    log.info("ingest.start")

    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job is None:
            # Not retryable -- the row doesn't exist, retrying changes nothing.
            log.error("ingest.job_not_found")
            return
        job.status = "ingesting"
        job.retry_count = job.retry_count + 1
        raw_object_key = job.raw_object_key

    tmp_dir = tempfile.mkdtemp(prefix=f"ingest-{stream_job_id}-")
    try:
        local_video_path = storage.get_local_path(
            raw_object_key, download_to=os.path.join(tmp_dir, "source")
        )

        try:
            duration_seconds = _probe_duration_seconds(local_video_path)
        except PermanentIngestError as exc:
            _mark_failed(stream_job_id, str(exc))
            log.error("ingest.permanent_failure", error=str(exc))
            return  # do not raise -- raising would trigger RQ retry on a permanent error

        local_audio_path = os.path.join(tmp_dir, "audio.wav")
        _extract_audio(local_video_path, local_audio_path)
        storage.put_file(local_audio_path, audio_object_key(stream_job_id))

        with db_session() as db:
            job = db.get(StreamJob, uuid.UUID(stream_job_id))
            job.duration_seconds = duration_seconds
            job.status = "ingested"

        log.info("ingest.done", duration_seconds=duration_seconds)

        # multiplier=8: local CPU Whisper (the default provider) is the
        # slowest realistic path here and the one that actually timed out
        # in practice on the user's machine at RQ's 180s default -- 8x is
        # deliberately generous (worst-case slow/loaded hardware, not
        # average-case) rather than tuned to look good on this sandbox's
        # fast synthetic test clips. buffer=600 covers the one-time model
        # download on a fresh machine, not just per-job fixed overhead.
        transcription_timeout = estimate_job_timeout_seconds(
            duration_seconds, multiplier=8.0, minimum=600, buffer=600
        )
        enqueue(
            "transcription", transcription.run, stream_job_id,
            on_failure=transcription.on_failure, job_timeout=transcription_timeout,
        )

    finally:
        for name in ("source", "audio.wav"):
            path = os.path.join(tmp_dir, name)
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


def _mark_failed(stream_job_id: str, error: str) -> None:
    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job:
            job.status = "failed_ingest"
            job.last_error = error[:4000]


def on_failure(job, connection, type, value, traceback) -> None:
    """RQ failure callback -- fires once retries are exhausted (see app.core.queue.enqueue)."""
    stream_job_id = job.args[0]
    _mark_failed(stream_job_id, f"{type.__name__}: {value}")
    logger.bind(stream_job_id=stream_job_id, worker="ingest").error(
        "ingest.retries_exhausted", error=str(value)
    )
