"""stream_jobs.camera_layout_mode -- per-job override for reaction-split
layout detection

Adds an explicit per-job override for app.core.rendering_logic
.classify_reaction_layout's auto-detection, so a creator can force "never
split" (a VOD with no facecam at all, where a false-positive detection
would otherwise steal screen space from the actual content) or "always
split when a face is found" (a VOD the creator knows has a facecam, when
the classifier's fixed size/corner thresholds guess wrong for their
specific webcam) instead of only ever getting the automatic heuristic. See
app.db.models.StreamJob.camera_layout_mode and app/workers/rendering.py.

Revision ID: c3d4e5f6a7b9
Revises: b2c3d4e5f6a8
Create Date: 2026-08-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'c3d4e5f6a7b9'
down_revision: Union[str, None] = 'b2c3d4e5f6a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('stream_jobs', sa.Column('camera_layout_mode', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('stream_jobs', 'camera_layout_mode')
