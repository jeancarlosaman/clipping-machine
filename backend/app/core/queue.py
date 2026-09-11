"""Redis + RQ queue wiring -- one named queue per pipeline stage.

Default per architecture doc §8 trade-off table: Redis + RQ over
Postgres-as-queue, for simple built-in retry/backoff semantics without much
extra ops burden. If that trade-off flips later, this is the one module
that needs to change -- API and worker code only ever call `enqueue(...)`.
"""
from __future__ import annotations

from redis import Redis
from rq import Queue, Retry

from app.core.config import settings

_redis_conn = Redis.from_url(settings.redis_url)

# One queue per pipeline stage keeps failure/backpressure isolated per stage
# (a stuck rendering queue shouldn't starve ingest) and lets `rq worker`
# processes subscribe to only the stages they run.
STAGE_QUEUES = (
    "ingest",
    "transcription",
    "segmentation",
    "scoring",
    "rendering",
    "captions",  # app.workers.caption_generation -- LLM-assisted post caption/hashtags, enqueued by rendering.py
    "upload",
)

_queues: dict[str, Queue] = {name: Queue(name, connection=_redis_conn) for name in STAGE_QUEUES}


def get_queue(stage: str) -> Queue:
    if stage not in _queues:
        raise ValueError(f"Unknown queue stage '{stage}'. Known stages: {sorted(_queues)}")
    return _queues[stage]


def enqueue(stage: str, func, *args, max_retries: int = 3, on_failure=None, **kwargs):
    """Enqueue `func(*args, **kwargs)` onto the named stage queue.

    max_retries uses RQ's built-in exponential-backoff Retry -- see
    architecture doc §7 for per-worker retry counts; pass max_retries=0 for
    workers where retrying is not meaningful (e.g. permanent 4xx errors are
    handled inside the job function instead, per-worker, not here). RQ's
    own Retry object rejects max=0 outright (ValueError), so that case is
    handled here as "no retry object at all" rather than passed through.

    on_failure, if given, is an RQ failure callback (job, connection, type,
    value, traceback) -> None, called once retries are exhausted -- every
    stage uses this to flip its row to its failed_* status. Must be an
    importable function (RQ serializes it by reference), not a lambda/closure.
    """
    queue = get_queue(stage)
    retry = Retry(max=max_retries, interval=[10, 30, 60][:max_retries] or [10]) if max_retries > 0 else None
    return queue.enqueue(
        func,
        *args,
        retry=retry,
        on_failure=on_failure,
        **kwargs,
    )


def redis_connection() -> Redis:
    return _redis_conn
