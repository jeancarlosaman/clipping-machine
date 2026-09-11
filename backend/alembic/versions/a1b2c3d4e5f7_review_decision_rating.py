"""review_decisions.rating -- optional 1..5 reviewer quality rating

Adds an optional subjective quality rating (1..5) a reviewer can attach to
a review decision, separate from the existing approve/reject/skip gate --
see app.db.models.ReviewDecision.rating's docstring for why these are kept
distinct, and README's "Reviewer feedback & ratings" for the intended use
(a manual input for sanity-checking SCORE_WEIGHT_* later, not automatic
retraining).

Revision ID: a1b2c3d4e5f7
Revises: f1a2b3c4d5e6
Create Date: 2026-08-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('review_decisions', sa.Column('rating', sa.SmallInteger(), nullable=True))
    op.create_check_constraint(
        'ck_review_decisions_rating_range',
        'review_decisions',
        'rating IS NULL OR (rating >= 1 AND rating <= 5)',
    )


def downgrade() -> None:
    op.drop_constraint('ck_review_decisions_rating_range', 'review_decisions', type_='check')
    op.drop_column('review_decisions', 'rating')
