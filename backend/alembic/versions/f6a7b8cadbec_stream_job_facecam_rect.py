"""stream_jobs.facecam_rect -- hand-marked facecam box

Revision ID: f6a7b8cadbec
Revises: e5f6a7b8cadb
Create Date: 2026-09-11

Normalized {x, y, w, h} (fractions of the source frame) marked by the
creator on a real frame of their own VOD. Nullable: null means "no manual
mark, detect a face like before", which is every pre-existing row.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f6a7b8cadbec"
down_revision: Union[str, None] = "e5f6a7b8cadb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("stream_jobs", sa.Column("facecam_rect", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("stream_jobs", "facecam_rect")
