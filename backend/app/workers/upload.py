"""Upload worker -- architecture doc §7. Real TikTok Content Posting API
integration (inbox/draft flow), not a stub.

Trigger:  creator action, POST /api/v1/clips/{clip_id}/upload (creates the
          upload_tasks row; this worker is enqueued right after)
Input:    upload_task_id
Output:   TikTok post id (task.external_post_id), upload_audit_log rows (one
          per attempt, regardless of outcome -- a hard product requirement)
State:    queued -> uploading -> uploaded | failed | blocked
Retries:  RQ retries the whole `run()` up to 3x with backoff (10/30/60s, see
          app.core.queue.enqueue's default) on anything this module treats
          as *retryable* -- network errors, timeouts, TikTok 5xx, or "still
          processing" after this attempt's polling budget runs out. Nothing
          else re-raises: a blocked task (cap/confidence/target_mode), a
          missing account/clip, or a TikTok 4xx/policy rejection is marked
          `failed` (or `blocked`) in place and `run()` returns normally, so
          RQ does not retry something a retry can't fix.

Idempotency across retries: a retry re-runs `run()` from the top, which
would normally mean re-uploading the whole video and creating a *second*,
orphaned draft on TikTok's side if the first attempt got past init but
failed later (e.g. mid-chunk network blip, or the polling budget running
out while TikTok is still processing). `upload_tasks.publish_id` is the
guard against that: it's persisted immediately once TikTok's init call
succeeds, and `run()` checks it first -- if already set, it skips straight
to polling that publish_id's status instead of calling init again. See that
column's comment on the model for the same point from the schema side.

Blocking note: status polling happens inline via `time.sleep` between
calls, tying up one worker slot for up to
`tiktok_status_poll_max_attempts * tiktok_status_poll_interval_seconds`
(~1 minute by default) per attempt. Fine at MVP's one-creator/one-replica
scale (same trade-off the architecture doc already accepts for the other
synchronous workers); a queue-depth problem here is the signal to move
status checking into its own polling job instead of inlining it, not
something to preemptively build.

Needs (see .env.example): TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET,
TIKTOK_REDIRECT_URI (all consumed by app.core.tiktok_client and
app.api.routers.creator_accounts' OAuth endpoints, not here directly --
this module only ever sees an already-issued access token off the
CreatorAccount row, refreshing it first via that same client if it's
expired/near-expiry and a refresh_token is on file).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone

from app.core import tiktok_client
from app.core.config import settings
from app.core.crypto import decrypt_token, encrypt_token
from app.core.storage import storage
from app.core.tiktok_client import TikTokAPIError
from app.db.models import CreatorAccount, RenderedClip, UploadAuditLog, UploadTask
from app.workers.common import db_session, logger

# Refresh proactively if the stored token expires within this window,
# rather than waiting for TikTok to actually reject a call with it --
# avoids burning an upload attempt (and its audit-log "failure" row) on a
# race between "token still valid when we checked" and "token expired by
# the time the request landed."
_TOKEN_REFRESH_SKEW = timedelta(minutes=5)


def _daily_cap_reached(db, creator_account_id: uuid.UUID, cap: int) -> bool:
    since = datetime.now(timezone.utc) - timedelta(days=1)
    count = (
        db.query(UploadTask)
        .filter(
            UploadTask.creator_account_id == creator_account_id,
            UploadTask.status == "uploaded",
            UploadTask.created_at >= since,
        )
        .count()
    )
    return count >= cap


def _mark_failed(task_id: uuid.UUID, message: str) -> None:
    """Terminal, non-retryable failure. Opens its own session -- callers
    are past the original db_session block by the time they know this
    (e.g. after a TikTokAPIError raised out of an HTTP call)."""
    with db_session() as db:
        task = db.get(UploadTask, task_id)
        if task is None:
            return
        task.status = "failed"
        task.last_error = message[:4000]
        db.add(UploadAuditLog(upload_task_id=task.id, event="failure", detail={"reason": message[:2000]}))


def _get_valid_access_token(creator_account_id: uuid.UUID, log) -> str:
    """Returns a usable access token for this account, refreshing and
    persisting a new one first if the stored one is expired/near-expiry
    and a refresh_token is on file. If there's no refresh_token to use
    (e.g. a manually-pasted token, which never expires by our own
    knowledge), just returns the stored token as-is and lets the actual
    API call surface the real error rather than guessing wrong here.
    """
    with db_session() as db:
        account = db.get(CreatorAccount, creator_account_id)
        if account is None:
            raise TikTokAPIError("creator account no longer exists", retryable=False)
        access_token = decrypt_token(account.access_token_encrypted)
        needs_refresh = (
            account.token_expires_at is not None
            and account.token_expires_at <= datetime.now(timezone.utc) + _TOKEN_REFRESH_SKEW
        )
        refresh_token_encrypted = account.refresh_token_encrypted

    if not needs_refresh or not refresh_token_encrypted:
        return access_token

    token_data = tiktok_client.refresh_access_token(decrypt_token(refresh_token_encrypted))
    new_access_token = token_data.get("access_token")
    if not new_access_token:
        raise TikTokAPIError("token refresh response missing access_token", retryable=False)

    with db_session() as db:
        account = db.get(CreatorAccount, creator_account_id)
        if account is not None:
            account.access_token_encrypted = encrypt_token(new_access_token)
            new_refresh_token = token_data.get("refresh_token")
            if new_refresh_token:
                account.refresh_token_encrypted = encrypt_token(new_refresh_token)
            expires_in = token_data.get("expires_in")
            account.token_expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=int(expires_in)) if expires_in else None
            )
    log.info("upload.token_refreshed")
    return new_access_token


def _upload_and_publish(
    task_id: uuid.UUID, creator_account_id: uuid.UUID, video_key: str, existing_publish_id: str | None, log
) -> tuple[str, dict]:
    """Init (skipped if existing_publish_id is already set -- see module
    docstring on idempotency) + chunked upload + bounded status polling.
    Returns (status, status_data) once TikTok reports a terminal status.
    Raises TikTokAPIError (retryable or not, per that error's own
    classification) for anything that didn't reach a terminal status.
    """
    access_token = _get_valid_access_token(creator_account_id, log)
    publish_id = existing_publish_id

    if publish_id is None:
        tmp_dir = tempfile.mkdtemp(prefix=f"upload-{task_id}-")
        try:
            local_path = storage.get_local_path(video_key, download_to=os.path.join(tmp_dir, "clip.mp4"))
            video_size = os.path.getsize(local_path)
            chunk_size, total_chunks = tiktok_client.plan_chunks(
                video_size, preferred_chunk_size=settings.tiktok_upload_chunk_size_bytes
            )
            init_data = tiktok_client.init_inbox_upload(
                access_token, video_size=video_size, chunk_size=chunk_size, total_chunk_count=total_chunks
            )
            publish_id = init_data["publish_id"]
            upload_url = init_data["upload_url"]

            # Persist publish_id immediately, before sending any bytes --
            # a crash/timeout anywhere below this point must not cause a
            # retry to call init() again (see module docstring).
            with db_session() as db:
                t = db.get(UploadTask, task_id)
                if t is not None:
                    t.publish_id = publish_id
            log.info("upload.initiated", publish_id=publish_id, video_size=video_size, total_chunks=total_chunks)

            with open(local_path, "rb") as f:
                offset = 0
                for _ in range(total_chunks):
                    this_chunk_size = min(chunk_size, video_size - offset)
                    chunk_bytes = f.read(this_chunk_size)
                    tiktok_client.upload_video_chunk(
                        upload_url,
                        chunk_bytes,
                        first_byte=offset,
                        last_byte=offset + this_chunk_size - 1,
                        total_bytes=video_size,
                    )
                    offset += this_chunk_size
            log.info("upload.bytes_sent", publish_id=publish_id)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        log.info("upload.resuming_existing_publish", publish_id=publish_id)

    for attempt in range(settings.tiktok_status_poll_max_attempts):
        status_data = tiktok_client.fetch_publish_status(access_token, publish_id)
        status = status_data.get("status")
        log.info("upload.status_poll", publish_id=publish_id, status=status, attempt=attempt)
        if status in tiktok_client.TERMINAL_SUCCESS_STATUSES or status == tiktok_client.TERMINAL_FAILURE_STATUS:
            return status, status_data
        time.sleep(settings.tiktok_status_poll_interval_seconds)

    # Still processing after this attempt's polling budget -- not a
    # failure, just unfinished. Retryable: the next attempt resumes
    # polling this same publish_id (no re-upload, per the guard above).
    raise TikTokAPIError(
        f"still processing after {settings.tiktok_status_poll_max_attempts} status checks", retryable=True
    )


def run(upload_task_id: str) -> None:
    log = logger.bind(upload_task_id=upload_task_id, worker="upload")
    log.info("upload.start")

    with db_session() as db:
        task = db.get(UploadTask, uuid.UUID(upload_task_id))
        if task is None:
            log.error("upload.task_not_found")
            return

        db.add(UploadAuditLog(upload_task_id=task.id, event="attempt", detail={"target_mode": task.target_mode}))

        # Hard product limits (architecture doc §6) are enforced here,
        # before any external call is made.
        if task.target_mode != "draft":
            task.status = "blocked"
            task.block_reason = "direct_publish_not_supported"
            db.add(UploadAuditLog(upload_task_id=task.id, event="blocked", detail={"reason": task.block_reason}))
            log.warning("upload.blocked", reason=task.block_reason)
            return

        # Cap is per-account (creator_accounts.daily_upload_cap), not the
        # global config default -- the config value is only the default
        # applied when an account is created, per architecture doc §5.
        account = db.get(CreatorAccount, task.creator_account_id)
        cap = account.daily_upload_cap if account else settings.default_daily_upload_cap
        if _daily_cap_reached(db, task.creator_account_id, cap):
            task.status = "blocked"
            task.block_reason = "daily_cap"
            db.add(UploadAuditLog(upload_task_id=task.id, event="blocked", detail={"reason": task.block_reason}))
            log.warning("upload.blocked", reason=task.block_reason)
            return

        if task.confidence_score is not None and task.confidence_score < settings.upload_confidence_threshold:
            task.status = "blocked"
            task.block_reason = "low_confidence"
            db.add(UploadAuditLog(upload_task_id=task.id, event="blocked", detail={"reason": task.block_reason}))
            log.warning("upload.blocked", reason=task.block_reason)
            return

        if account is None:
            task.status = "failed"
            task.last_error = "creator account no longer exists"
            db.add(UploadAuditLog(upload_task_id=task.id, event="failure", detail={"reason": task.last_error}))
            log.error("upload.account_missing")
            return

        clip = db.get(RenderedClip, task.rendered_clip_id)
        if clip is None or not clip.object_key:
            task.status = "failed"
            task.last_error = "rendered clip or its video file is missing"
            db.add(UploadAuditLog(upload_task_id=task.id, event="failure", detail={"reason": task.last_error}))
            log.error("upload.clip_missing")
            return

        task.status = "uploading"
        task.retry_count = task.retry_count + 1

        task_id = task.id
        creator_account_id = account.id
        video_key = clip.object_key
        existing_publish_id = task.publish_id

    try:
        status, status_data = _upload_and_publish(task_id, creator_account_id, video_key, existing_publish_id, log)
    except TikTokAPIError as exc:
        if exc.retryable:
            log.warning("upload.retryable_error", error=str(exc))
            raise  # RQ's Retry re-enqueues; on_failure only fires once retries are exhausted
        _mark_failed(task_id, str(exc))
        log.error("upload.permanent_failure", error=str(exc))
        return

    if status in tiktok_client.TERMINAL_SUCCESS_STATUSES:
        with db_session() as db:
            task = db.get(UploadTask, task_id)
            if task is not None:
                task.status = "uploaded"
                task.last_error = None
                post_ids = status_data.get("publicaly_available_post_id") or []
                task.external_post_id = str(post_ids[0]) if post_ids else task.publish_id
                db.add(UploadAuditLog(upload_task_id=task.id, event="success", detail={"status": status}))
        log.info("upload.done", status=status)
        return

    # status == FAILED -- terminal per TikTok, no retry would help.
    fail_reason = status_data.get("fail_reason") or "unknown"
    _mark_failed(task_id, f"TikTok publish failed: {fail_reason}")
    log.error("upload.tiktok_reported_failure", fail_reason=fail_reason)


def on_failure(job, connection, type, value, traceback) -> None:
    upload_task_id = job.args[0] if job.args else None
    if not upload_task_id:
        return
    with db_session() as db:
        task = db.get(UploadTask, uuid.UUID(upload_task_id))
        if task:
            task.status = "failed"
            task.last_error = f"{type.__name__}: {value}"[:4000]
            db.add(
                UploadAuditLog(
                    upload_task_id=task.id, event="failure", detail={"error": str(value)}
                )
            )
    logger.bind(upload_task_id=upload_task_id, worker="upload").error(
        "upload.retries_exhausted", error=str(value)
    )
