"""FastAPI app entrypoint. Run with: uvicorn app.main:app --reload"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_exception_handlers
from app.api.routers import clips, creator_accounts, stream_jobs, upload_tasks
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="Clipping Machine API",
    version="0.1.0",
    description="VOD-first AI clipping tool -- MVP API. See architecture doc for full design.",
)

register_exception_handlers(app)

app.include_router(stream_jobs.router)
app.include_router(clips.router)
app.include_router(upload_tasks.router)
app.include_router(creator_accounts.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


# Dev console (plain HTML/JS, no build step -- see web/app.js) -- mounted
# LAST and at "/" so it only catches requests that don't match an API route
# above. Same-origin, so the frontend needs no CORS config to call the API.
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
