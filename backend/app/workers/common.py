"""Shared helpers for worker code.

Workers run as separate processes (RQ), never inside a FastAPI request, so
they get their own DB session per job rather than using the `get_db`
FastAPI dependency.
"""
from __future__ import annotations

import subprocess
from contextlib import contextmanager
from decimal import Decimal

from app.core.logging import configure_logging, get_logger
from app.db.session import SessionLocal

configure_logging()
logger = get_logger(__name__)


@contextmanager
def db_session():
    """Session that commits on clean exit, rolls back and re-raises on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_subprocess(cmd: list[str], *, timeout_seconds: int = 1800) -> subprocess.CompletedProcess:
    """Run an external tool (ffmpeg/ffprobe/...) and raise with stderr on failure.

    Every worker that shells out should go through this so failures carry
    the tool's actual error output into last_error / logs, not just
    "non-zero exit".
    """
    logger.info("subprocess.start", cmd=cmd)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        logger.error("subprocess.failed", cmd=cmd, returncode=result.returncode, stderr=result.stderr[-4000:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr[-2000:]}")
    logger.info("subprocess.ok", cmd=cmd)
    return result


def estimate_job_timeout_seconds(
    duration_seconds: float | Decimal | None, *, multiplier: float, minimum: int, buffer: int = 120
) -> int:
    """A job timeout that scales with the video's own length instead of one
    fixed number for every job.

    RQ's own default job timeout is 180 seconds (see `rq.defaults`) -- fine
    for the short synthetic clips this pipeline was verified against during
    development, but real work here (CPU transcription, scene detection,
    ffmpeg encoding) takes time roughly proportional to how much video
    there is. A fixed timeout is either wastefully long for a 30s test clip
    or -- as happened on the user's own machine with a longer real upload --
    fails a perfectly healthy job partway through. See
    app.workers.transcription/segmentation/scoring/ingest for the
    per-stage `multiplier` reasoning; each is a deliberately generous
    (worst-case-hardware, not average-case) real-time factor.

    `buffer` covers fixed overhead unrelated to video length (model load,
    ffmpeg process startup, DB round trips). `minimum` is a floor so a very
    short clip doesn't get an unreasonably tight timeout.

    `duration_seconds` commonly arrives as a `decimal.Decimal` -- it's read
    back from a Postgres `Numeric` column (StreamJob.duration_seconds /
    CandidateSegment.start_seconds/end_seconds), and psycopg2 returns
    `Numeric` as `Decimal`, not `float`. `Decimal * float` raises
    TypeError -- a real bug hit in production on the user's machine on a
    real ~28-minute VOD (this function's first real caller with a duration
    that didn't come from a hand-constructed float in a test). Cast
    explicitly here once rather than trusting every call site to remember.
    """
    if not duration_seconds or duration_seconds <= 0:
        # Duration not known yet (e.g. the very first ingest enqueue, before
        # ffprobe has even run) -- fall back to a flat, generous ceiling
        # rather than RQ's 180s default.
        return max(minimum, 1800)
    return max(minimum, int(float(duration_seconds) * multiplier) + buffer)


def audio_object_key(stream_job_id: str) -> str:
    """Deterministic object key for a job's extracted audio track.

    Not stored as a column (the schema in the architecture doc doesn't have
    one) -- ingest and transcription both derive it from stream_job_id so no
    migration is needed for this internal convention. If a second audio
    format/variant is ever needed, that's the trigger to add a real column.
    """
    return f"audio/{stream_job_id}/audio.wav"
