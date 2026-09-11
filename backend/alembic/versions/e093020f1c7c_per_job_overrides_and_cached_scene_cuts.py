"""per-job overrides and cached scene cuts

Adds creator-configurable per-job overrides to stream_jobs (stt_model_size,
min_clip_seconds, max_clip_seconds, min_score_threshold -- all nullable,
null meaning "use the global settings.* default") plus scene_cuts_seconds,
which persists segmentation's PySceneDetect output so scoring can reuse it
instead of re-downloading the raw video and re-running scene detection from
scratch (see app/workers/scoring.py).

Revision ID: e093020f1c7c
Revises: 9aa9d6437404
Create Date: 2026-08-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'e093020f1c7c'
down_revision: Union[str, None] = '9aa9d6437404'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('stream_jobs', sa.Column('stt_model_size', sa.String(), nullable=True))
    op.add_column('stream_jobs', sa.Column('min_clip_seconds', sa.Numeric(), nullable=True))
    op.add_column('stream_jobs', sa.Column('max_clip_seconds', sa.Numeric(), nullable=True))
    op.add_column('stream_jobs', sa.Column('min_score_threshold', sa.Numeric(), nullable=True))
    op.add_column(
        'stream_jobs',
        sa.Column('scene_cuts_seconds', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('stream_jobs', 'scene_cuts_seconds')
    op.drop_column('stream_jobs', 'min_score_threshold')
    op.drop_column('stream_jobs', 'max_clip_seconds')
    op.drop_column('stream_jobs', 'min_clip_seconds')
    op.drop_column('stream_jobs', 'stt_model_size')
