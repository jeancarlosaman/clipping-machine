"""stream_jobs.style_overrides -- per-job burn-in styling

Revision ID: c9dbec012345
Revises: b8cadbec0123
Create Date: 2026-09-12

One JSONB column rather than five scalar ones: these are all knobs for the
same thing (how the burned-in layout looks), they are always set together
from the console's preview editor, and adding a sixth knob later should not
need another migration.

Null = use the settings.* defaults, which is every job predating this.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c9dbec012345"
down_revision: Union[str, None] = "b8cadbec0123"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("stream_jobs", sa.Column("style_overrides", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("stream_jobs", "style_overrides")
