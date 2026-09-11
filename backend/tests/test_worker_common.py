"""Pure unit tests for app.workers.common.estimate_job_timeout_seconds --
the fix for a real bug the user hit: RQ's default 180s job timeout killed
a real (non-test-fixture-length) transcription job partway through.
"""
from decimal import Decimal

from app.workers.common import estimate_job_timeout_seconds


def test_unknown_duration_falls_back_to_flat_generous_ceiling():
    assert estimate_job_timeout_seconds(None, multiplier=8.0, minimum=600) == 1800
    assert estimate_job_timeout_seconds(0, multiplier=8.0, minimum=600) == 1800
    assert estimate_job_timeout_seconds(-5, multiplier=8.0, minimum=600) == 1800


def test_scales_with_duration():
    # 600s (10min) video, 3x multiplier, 120s buffer (default).
    assert estimate_job_timeout_seconds(600, multiplier=3.0, minimum=300) == 600 * 3 + 120


def test_respects_explicit_buffer():
    assert estimate_job_timeout_seconds(600, multiplier=3.0, minimum=300, buffer=600) == 600 * 3 + 600


def test_short_duration_floors_at_minimum():
    # A 5s clip * 6x + 120s buffer is well under the minimum floor.
    assert estimate_job_timeout_seconds(5, multiplier=6.0, minimum=180) == 180


def test_long_duration_exceeds_rq_default_by_design():
    # This is the actual bug: RQ's default is 180s. A real ~20 minute VOD
    # transcribed at the ingest worker's multiplier=8 must clear that by a
    # wide margin, not brush up against it.
    twenty_minutes = 20 * 60
    timeout = estimate_job_timeout_seconds(twenty_minutes, multiplier=8.0, minimum=600, buffer=600)
    assert timeout > 180
    assert timeout == twenty_minutes * 8 + 600


def test_accepts_decimal_duration_from_db_numeric_column():
    # StreamJob.duration_seconds and CandidateSegment.start_seconds/
    # end_seconds are Postgres Numeric columns -- psycopg2 returns them as
    # decimal.Decimal, not float, when read back from the DB. This is the
    # actual bug hit on the user's machine on a real ~28-minute VOD:
    # transcription completed fine, then segmentation's enqueue crashed
    # with "unsupported operand type(s) for *: 'decimal.Decimal' and
    # 'float'" because duration_seconds came from job.duration_seconds
    # (a Decimal) and multiplier is a plain float.
    assert estimate_job_timeout_seconds(Decimal("600.0"), multiplier=3.0, minimum=300) == 600 * 3 + 120


def test_decimal_and_equivalent_float_produce_identical_timeout():
    decimal_result = estimate_job_timeout_seconds(Decimal("1680.5"), multiplier=3.0, minimum=300)
    float_result = estimate_job_timeout_seconds(1680.5, multiplier=3.0, minimum=300)
    assert decimal_result == float_result
