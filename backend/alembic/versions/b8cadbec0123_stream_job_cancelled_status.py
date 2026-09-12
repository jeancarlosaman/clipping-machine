"""stream_jobs.status gains 'cancelled'

Revision ID: b8cadbec0123
Revises: a7b8cadbec01
Create Date: 2026-09-12

Cooperative cancel: the API sets this status, and each worker checks it at
the top of its stage and stops instead of enqueuing the next one. Nothing
kills a running ffmpeg/whisper mid-frame, so a job stops at the next stage
boundary rather than instantly -- the trade is that nothing is ever left
half-written.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "b8cadbec0123"
down_revision: Union[str, None] = "a7b8cadbec01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PostgreSQL refuses ALTER TYPE ... ADD VALUE inside a transaction block,
    # and Alembic runs every migration in one. autocommit_block() steps
    # outside it for exactly this case. IF NOT EXISTS keeps the migration
    # re-runnable against a database where it was applied by hand.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE stream_job_status ADD VALUE IF NOT EXISTS 'cancelled'")


def downgrade() -> None:
    # Deliberately a no-op. PostgreSQL cannot drop a value from an enum type;
    # undoing this properly means recreating stream_job_status and rewriting
    # every column that uses it, which risks far more than an unused label
    # sitting in the type does. Downgrading past this leaves 'cancelled'
    # defined but unused, which is harmless.
    pass
