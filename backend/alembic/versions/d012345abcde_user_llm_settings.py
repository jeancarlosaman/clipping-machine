"""users.llm_provider + users.openai_api_key_encrypted

Revision ID: d012345abcde
Revises: c9dbec012345
Create Date: 2026-09-12

Per-account LLM configuration, so a creator can supply their own OpenAI key
and choose it over the locally-hosted Ollama without editing .env.

The key is stored encrypted with the same Fernet helper that protects
creator_accounts' TikTok tokens (app/core/crypto.py) -- it is a credential
that can spend the holder's money, so it never sits in the database as
plaintext and is never returned by the API.

Both nullable: null means "use the settings.* defaults", which is every
existing row.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d012345abcde"
down_revision: Union[str, None] = "c9dbec012345"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("llm_provider", sa.String(), nullable=True))
    op.add_column("users", sa.Column("openai_api_key_encrypted", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "openai_api_key_encrypted")
    op.drop_column("users", "llm_provider")
