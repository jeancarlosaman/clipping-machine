"""Transcription worker -- architecture doc §7.

Trigger:  stream_job.ingested (enqueued by app.workers.ingest.run)
Input:    stream_job_id (reads audio via app.workers.common.audio_object_key)
Output:   a `transcripts` row (segments with start/end/text, per architecture
          doc §5) -- this is what PySceneDetect-based segmentation and the
          scoring worker both read.
State:    ingested -> transcribing -> transcribed | failed_transcription
Retries:  3x with backoff on TransientSttError (network/rate-limit/5xx);
          PermanentSttError (bad audio, missing config) fails immediately
          without burning retries -- same pattern as ingest.py's
          PermanentIngestError.

Provider (open-source local Whisper by default, or the hosted OpenAI API)
is selected by STT_PROVIDER -- see app.core.stt. This module only talks to
the SttProvider interface, never a concrete provider, so switching providers
is a config change, not a code change here.
"""
from __future__ import annotations

import os
import tempfile
import uuid

from app.core.queue import enqueue
from app.core.storage import storage
from app.core.stt import PermanentSttError, TransientSttError, get_stt_provider
from app.db.models import StreamJob, Transcript
from app.workers import segmentation
from app.workers.common import audio_object_key, db_session, estimate_job_timeout_seconds, logger


def run(stream_job_id: str) -> None:
    log = logger.bind(stream_job_id=stream_job_id, worker="transcription")
    log.info("transcription.start")

    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job is None:
            # Not retryable -- the row doesn't exist, retrying changes nothing.
            log.error("transcription.job_not_found")
            return
        job.status = "transcribing"
        job.retry_count = job.retry_count + 1
        duration_seconds = job.duration_seconds  # captured now -- job is detached once this session closes
        stt_model_size = job.stt_model_size  # per-job override, see StreamJob.stt_model_size

    tmp_dir = tempfile.mkdtemp(prefix=f"transcribe-{stream_job_id}-")
    try:
        local_audio_path = storage.get_local_path(
            audio_object_key(stream_job_id), download_to=os.path.join(tmp_dir, "audio.wav")
        )

        provider = get_stt_provider(model_size=stt_model_size)
        log = log.bind(stt_provider=provider.name)
        try:
            result = provider.transcribe(local_audio_path)
        except PermanentSttError as exc:
            _mark_failed(stream_job_id, str(exc))
            log.error("transcription.permanent_failure", error=str(exc))
            return  # do not raise -- raising would trigger a pointless RQ retry
        except TransientSttError as exc:
            log.warning("transcription.transient_failure", error=str(exc))
            raise  # let RQ retry per the "transcription" queue's Retry policy

        with db_session() as db:
            db.add(
                Transcript(
                    stream_job_id=uuid.UUID(stream_job_id),
                    provider=provider.name,
                    language=result.language,
                    full_text=result.full_text,
                    segments=[
                        {"start": seg.start, "end": seg.end, "text": seg.text} for seg in result.segments
                    ],
                )
            )
            job = db.get(StreamJob, uuid.UUID(stream_job_id))
            job.status = "transcribed"

        log.info("transcription.done", segment_count=len(result.segments))

        # multiplier=3: PySceneDetect's content-aware scene detection reads
        # through the whole raw video once -- generally faster than
        # real-time, but 3x is a deliberately generous margin for slow
        # disks/CPUs rather than this sandbox's fast synthetic test clips.
        segmentation_timeout = estimate_job_timeout_seconds(duration_seconds, multiplier=3.0, minimum=300)

        # max_retries=2 to match segmentation.py's documented retry policy
        # (deterministic CPU work -- retries only cover transient I/O, not
        # logic errors that would just fail the same way again).
        enqueue(
            "segmentation", segmentation.run, stream_job_id, max_retries=2,
            on_failure=segmentation.on_failure, job_timeout=segmentation_timeout,
        )

    finally:
        audio_path = os.path.join(tmp_dir, "audio.wav")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


def _mark_failed(stream_job_id: str, error: str) -> None:
    with db_session() as db:
        job = db.get(StreamJob, uuid.UUID(stream_job_id))
        if job:
            job.status = "failed_transcription"
            job.last_error = error[:4000]


def on_failure(job, connection, type, value, traceback) -> None:
    """RQ failure callback -- fires once retries are exhausted (see app.core.queue.enqueue)."""
    stream_job_id = job.args[0]
    _mark_failed(stream_job_id, f"{type.__name__}: {value}")
    logger.bind(stream_job_id=stream_job_id, worker="transcription").error(
        "transcription.retries_exhausted", error=str(value)
    )
