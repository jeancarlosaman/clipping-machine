"""stream_jobs.crop_bias -- per-job override for which side of the frame
the 9:16 crop favours

Companion to the "fit_frame" camera_layout_mode value added in the same
change (that one needs no migration -- camera_layout_mode is a plain
String column, not a native Postgres enum, so a new allowed value is an
application-level validation change only). This column is the escape
hatch for creators who still want a crop but whose important content
consistently sits off-center: "left"/"center"/"right", null = today's
centered/face-following behavior. See app.db.models.StreamJob.crop_bias
and app.core.rendering_logic.compute_crop_offset.

Revision ID: d4e5f6a7b8ca
Revises: c3d4e5f6a7b9
Create Date: 2026-09-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'd4e5f6a7b8ca'
down_revision: Union[str, None] = 'c3d4e5f6a7b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('stream_jobs', sa.Column('crop_bias', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('stream_jobs', 'crop_bias')
