"""upload_tasks.publish_id for idempotent TikTok upload retries

Adds `publish_id` (TikTok's own id for an in-progress inbox upload,
returned by /v2/post/publish/inbox/video/init/) so app.workers.upload.run()
can tell "bytes already uploaded, just resume polling status" apart from
"never attempted" on retry -- see that worker's module docstring.

Revision ID: f1a2b3c4d5e6
Revises: e093020f1c7c
Create Date: 2026-08-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e093020f1c7c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('upload_tasks', sa.Column('publish_id', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('upload_tasks', 'publish_id')
