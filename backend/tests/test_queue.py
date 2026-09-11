"""Unit test for app.core.queue.enqueue's max_retries=0 handling -- a real
bug caught while wiring up app.workers.caption_generation (RQ's Retry
object raises ValueError for max=0, so max_retries=0 must skip building one
entirely rather than passing max=0 through to Retry()). Uses the real
Redis test instance (see tests/conftest.py) rather than mocking rq.Queue,
since the whole point is to prove `queue.enqueue(...)` doesn't raise.
"""
from app.core.queue import enqueue


def _noop(*args, **kwargs):
    pass


def test_enqueue_with_max_retries_zero_does_not_raise():
    job = enqueue("rendering", _noop, max_retries=0)
    assert job is not None


def test_enqueue_with_positive_max_retries_still_works():
    job = enqueue("rendering", _noop, max_retries=2)
    assert job is not None
