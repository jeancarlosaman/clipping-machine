"""GET /api/v1/upload-tasks/{id} -- architecture doc §6."""
from __future__ import annotations

import uuid

from fastapi import APIRouter
from sqlalchemy.orm import Session

from app.api.deps import CurrentUserDep, DbDep
from app.api.errors import ApiError
from app.db.models import UploadTask, User
from app.schemas import UploadTaskOut

router = APIRouter(prefix="/api/v1/upload-tasks", tags=["upload-tasks"])


@router.get("/{upload_task_id}", response_model=UploadTaskOut)
def get_upload_task(upload_task_id: uuid.UUID, db: Session = DbDep, user: User = CurrentUserDep) -> UploadTask:
    task = db.get(UploadTask, upload_task_id)
    if task is None:
        raise ApiError(404, "not_found", "Upload task not found")
    if task.rendered_clip.stream_job.user_id != user.id:
        raise ApiError(403, "forbidden", "You do not have access to this upload task")
    return task
