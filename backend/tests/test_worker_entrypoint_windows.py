"""Regression test for the Windows registry-cleanup crash hit in a real dev
session: `python worker_entrypoint.py` crashed on startup (inside RQ's
periodic clean_registries() maintenance, sweeping a leftover job from a
prior run) with `AttributeError: module 'signal' has no attribute
'SIGALRM'`. worker_entrypoint.py already patched RQ's Worker class to avoid
that exact error for the worker's own job-timeout handling
(_WindowsSimpleWorker), but rq.registry.BaseRegistry has its own, separate
death_penalty_class class attribute that clean_registries() uses instead --
untouched by that first patch, since registries are constructed directly by
RQ's own code, not via our Worker subclass. This test asserts BOTH patches
land on sys.platform == "win32", so a regression here (e.g. someone
"simplifying" the Worker patch back down) would fail loudly in CI instead
of only on a real Windows machine.

This sandbox runs Linux, so these tests force the Windows branch by
monkeypatching sys.platform and reloading the module -- they can't launch a
real worker process on Windows, but they do exercise the exact class
attributes the crash traceback pointed at.
"""
import importlib
import sys

from rq.registry import BaseRegistry
from rq.timeouts import TimerDeathPenalty
from rq.worker import Worker

import worker_entrypoint


def test_non_windows_leaves_rq_defaults_untouched():
    # Baseline for the platform this repo's own test suite actually runs on
    # (Linux) -- confirms the "else" branch is a real no-op, so the
    # Windows-branch assertions below are meaningful rather than vacuously
    # true because nothing ever un-patches itself.
    assert worker_entrypoint._WORKER_CLASS is Worker


def test_windows_platform_patches_worker_and_registry_death_penalty():
    # Both sys.platform and BaseRegistry.death_penalty_class are restored by
    # hand rather than via pytest's monkeypatch fixture: reload() makes the
    # module's patching a real, persistent side effect (not fixture-scoped),
    # and monkeypatch's own teardown only runs after this function returns
    # -- too late for the restoring reload below, which needs
    # sys.platform back to its real value *before* it re-executes the module.
    original_platform = sys.platform
    original_registry_penalty = BaseRegistry.death_penalty_class
    try:
        sys.platform = "win32"
        reloaded = importlib.reload(worker_entrypoint)

        assert reloaded._WORKER_CLASS is reloaded._WindowsSimpleWorker
        assert reloaded._WORKER_CLASS.death_penalty_class is TimerDeathPenalty

        # This is the assertion that would have caught the actual bug: a
        # SEPARATE class attribute, on rq.registry.BaseRegistry (used by
        # RQ's clean_registries() maintenance sweep, not the worker's own
        # job-execution path), also has to be patched -- fixing only
        # _WORKER_CLASS above leaves this one still pointing at the
        # Unix-only default.
        assert BaseRegistry.death_penalty_class is TimerDeathPenalty
    finally:
        sys.platform = original_platform
        BaseRegistry.death_penalty_class = original_registry_penalty
        importlib.reload(worker_entrypoint)
        assert worker_entrypoint._WORKER_CLASS is Worker
