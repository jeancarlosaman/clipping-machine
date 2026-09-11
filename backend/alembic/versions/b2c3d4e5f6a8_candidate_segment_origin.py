"""candidate_segments.origin/llm_reason -- LLM segment suggestions

Adds the two columns app.core.llm_segmentation needs to record which
candidate windows came from the deterministic sliding-window pass
("heuristic", the default -- every row before this migration) versus the
new opt-in LLM candidate-proposer ("llm"), plus that proposer's stated
reason for suggesting the window. See app.db.models.CandidateSegment
.origin/.llm_reason and app/workers/segmentation.py for how both origins
land in the same pending_score pool.

Revision ID: b2c3d4e5f6a8
Revises: a1b2c3d4e5f7
Create Date: 2026-08-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b2c3d4e5f6a8'
down_revision: Union[str, None] = 'a1b2c3d4e5f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'candidate_segments',
        sa.Column('origin', sa.String(), nullable=False, server_default='heuristic'),
    )
    op.add_column('candidate_segments', sa.Column('llm_reason', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('candidate_segments', 'llm_reason')
    op.drop_column('candidate_segments', 'origin')
