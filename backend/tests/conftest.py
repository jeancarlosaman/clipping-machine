"""Pytest fixtures.

Points the app at a dedicated test Postgres database and an isolated Redis
DB index (15) *before* any app module is imported, so app.core.config's
cached Settings singleton and app.db.session's engine pick up test values
instead of the dev ones. Requires Postgres + Redis reachable exactly like
local dev (see README) -- this suite talks to real services, it does not
mock the database or queue.
"""
import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://clipping_machine:clipping_machine@localhost:5432/clipping_machine_test",
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("STORAGE_BACKEND", "local")

import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ["STORAGE_LOCAL_DIR"] = tempfile.mkdtemp(prefix="clipping-machine-test-storage-")

from app.core.auth import create_access_token  # noqa: E402
from app.db.models import Base, User  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    shutil.rmtree(os.environ["STORAGE_LOCAL_DIR"], ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_tables():
    """Truncate all tables between tests so they don't leak state into each other."""
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def user(db_session):
    u = User(email=f"{uuid.uuid4()}@example.com", password_hash="test", display_name="Test User")
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    return u


@pytest.fixture
def auth_headers(user):
    token = create_access_token(str(user.id))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_video_path(tmp_path):
    """A tiny real MP4, generated with ffmpeg, for upload/ingest tests.

    Skips (rather than fabricating a fake video) if ffmpeg isn't on PATH --
    ffmpeg is a hard project dependency (see architecture doc), so a missing
    binary should be visible as a skip reason, not a silent pass.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    path = tmp_path / "sample.mp4"
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture
def three_scene_video_path(tmp_path):
    """A 9s real MP4 with three unambiguous scene cuts (red/blue/green, 3s
    each) at 3.0s and 6.0s -- for segmentation-worker tests that need
    PySceneDetect to find real, predictable cuts rather than a video too
    short/uniform to have any.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    path = tmp_path / "three_scene.mp4"
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=red:size=320x240:duration=3:rate=10",
            "-f", "lavfi", "-i", "color=c=blue:size=320x240:duration=3:rate=10",
            "-f", "lavfi", "-i", "color=c=green:size=320x240:duration=3:rate=10",
            "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[out]",
            "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path
