"""candidate_segments.parts -- optional multi-part ("stitched") clips

A moment worth clipping isn't always one contiguous stretch of a VOD: a
story can be told in two passes, or a punchline can call back to something
said ten minutes earlier. This column lets one candidate describe several
non-adjacent [start, end] pairs that get cut out and joined end-to-end at
render time.

Nullable, and null for every candidate that existed before this migration
(and for every heuristic candidate today) -- those remain plain contiguous
windows described by start_seconds/end_seconds, which stay populated for
multi-part candidates too so existing queries/ordering/overlap checks keep
working unchanged. See app.db.models.CandidateSegment.parts and
app.core.rendering_logic.normalize_parts.

Revision ID: e5f6a7b8cadb
Revises: d4e5f6a7b8ca
Create Date: 2026-09-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'e5f6a7b8cadb'
down_revision: Union[str, None] = 'd4e5f6a7b8ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'candidate_segments',
        sa.Column('parts', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('candidate_segments', 'parts')
