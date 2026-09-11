"""stream_jobs.gameplay_rect -- hand-marked content region

Revision ID: a7b8cadbec01
Revises: f6a7b8cadbec
Create Date: 2026-09-11

The companion to facecam_rect: which part of the source frame is the actual
content. Drives the split layout's bottom panel, and the crop region for a
single-crop layout. Nullable; null keeps the previous centered/face-driven
behaviour.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a7b8cadbec01"
down_revision: Union[str, None] = "f6a7b8cadbec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("stream_jobs", sa.Column("gameplay_rect", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("stream_jobs", "gameplay_rect")
