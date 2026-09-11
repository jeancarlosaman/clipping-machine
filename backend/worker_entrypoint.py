"""Run an RQ worker listening on all pipeline stage queues.

Usage:
    python worker_entrypoint.py                     # listen on every stage
    python worker_entrypoint.py ingest transcription # listen on specific stages only

In production, run separate worker processes/replicas per stage (or per
group of stages) so one stage's backlog can't starve another -- see
architecture doc §4/§7. Splitting queues per process is a one-line change:
`python worker_entrypoint.py rendering`.
"""
from __future__ import annotations

import sys

from rq.registry import BaseRegistry
from rq.timeouts import TimerDeathPenalty
from rq.worker import SimpleWorker, Worker

from app.core.logging import configure_logging, get_logger
from app.core.queue import STAGE_QUEUES, get_queue, redis_connection

configure_logging()
logger = get_logger(__name__)


class _WindowsSimpleWorker(SimpleWorker):
    """SimpleWorker (see below) still enforces job timeouts via
    UnixSignalDeathPenalty by default, which needs signal.SIGALRM -- also
    Unix-only, and the second Windows-incompatible thing RQ does by
    default (the first is Worker's os.fork() per job). TimerDeathPenalty is
    RQ's own threading+ctypes based equivalent, which works on Windows.
    """

    death_penalty_class = TimerDeathPenalty


# RQ's default Worker runs each job in a forked subprocess (os.fork()) so a
# hung/crashed job can't take the worker process down with it. os.fork()
# doesn't exist on Windows at all, so the default Worker crashes on the
# very first job there (AttributeError: module 'os' has no attribute
# 'fork') -- _WindowsSimpleWorker runs jobs in-process instead (via
# SimpleWorker) with a Windows-compatible timeout mechanism (see above).
# Trade-off: on Windows a job that hangs forever is killed by a Python-level
# async exception instead of a forked-process kill -- less bulletproof than
# Linux/macOS, but correct and sufficient for local dev.
_WORKER_CLASS = _WindowsSimpleWorker if sys.platform == "win32" else Worker

if sys.platform == "win32":
    # A THIRD Windows-incompatible thing RQ does by default, found the hard
    # way (a Windows worker crashing on startup while sweeping a leftover
    # job from a prior run): periodic registry maintenance (clean_registries,
    # called from Worker.run_maintenance_tasks on every startup and
    # periodically after) replays a stale job's failure callback through
    # rq.registry.BaseRegistry's OWN death_penalty_class -- a separate class
    # attribute from Worker's (patched above), hardcoded to
    # UnixSignalDeathPenalty and never touched by _WindowsSimpleWorker. Every
    # registry subclass (StartedJobRegistry, FailedJobRegistry, etc.)
    # inherits this one shared attribute rather than setting its own, so
    # patching it here once, before any registry is constructed, covers all
    # of them. Without this, a Windows worker can crash on startup or during
    # routine maintenance any time a job needing failure-callback replay
    # (e.g. one abandoned by a prior crashed worker) is swept up -- not just
    # during normal job execution, which is why _WORKER_CLASS's own override
    # above didn't catch it.
    BaseRegistry.death_penalty_class = TimerDeathPenalty


def main() -> None:
    stages = sys.argv[1:] or list(STAGE_QUEUES)
    unknown = set(stages) - set(STAGE_QUEUES)
    if unknown:
        raise SystemExit(f"Unknown stage(s): {sorted(unknown)}. Known: {sorted(STAGE_QUEUES)}")

    queues = [get_queue(stage) for stage in stages]
    logger.info("worker.start", stages=stages, worker_class=_WORKER_CLASS.__name__)
    worker = _WORKER_CLASS(queues, connection=redis_connection())
    worker.work()


if __name__ == "__main__":
    main()
