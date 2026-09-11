"""Object storage abstraction.

Per architecture doc §8: S3-compatible storage is the default, local disk is
a dev-only fallback. Both implementations share this interface so workers
and API code never branch on backend -- they just call `storage.put_file` /
`storage.get_local_path` and the config decides what actually happens.

NOTE: this is intentionally minimal (put/get/exists), not a general-purpose
filesystem shim. Add methods only when a real caller needs them.
"""
from __future__ import annotations

import abc
import os
import shutil
import uuid
from pathlib import Path

from app.core.config import settings


def new_object_key(prefix: str, extension: str) -> str:
    """Generate a collision-safe object key, e.g. 'raw/9c1e.../source.mp4'."""
    extension = extension.lstrip(".")
    return f"{prefix.strip('/')}/{uuid.uuid4()}.{extension}"


class ObjectStorage(abc.ABC):
    @abc.abstractmethod
    def put_file(self, local_path: str, object_key: str) -> None:
        """Upload/copy a local file into storage at `object_key`."""

    @abc.abstractmethod
    def get_local_path(self, object_key: str, download_to: str) -> str:
        """Ensure `object_key` is available as a local file, return its path.

        For local storage this is a no-copy passthrough where possible; for
        S3 this downloads to `download_to`. Callers (e.g. ffmpeg workers)
        always need a real local path, never a stream.
        """

    @abc.abstractmethod
    def exists(self, object_key: str) -> bool:
        ...

    @abc.abstractmethod
    def delete(self, object_key: str) -> None:
        """Remove `object_key` from storage. Must not raise if it's already
        gone -- callers use this for cleanup (e.g. deleting a stream_job's
        files), where "already deleted" is a success, not an error."""


class LocalFilesystemStorage(ObjectStorage):
    """Dev-only backend. Do not use across multiple machines/replicas."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, object_key: str) -> Path:
        path = (self.root_dir / object_key).resolve()
        if self.root_dir.resolve() not in path.parents and path != self.root_dir.resolve():
            raise ValueError(f"object_key escapes storage root: {object_key}")
        return path

    def put_file(self, local_path: str, object_key: str) -> None:
        dest = self._resolve(object_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)

    def get_local_path(self, object_key: str, download_to: str) -> str:
        return str(self._resolve(object_key))

    def exists(self, object_key: str) -> bool:
        return self._resolve(object_key).exists()

    def delete(self, object_key: str) -> None:
        self._resolve(object_key).unlink(missing_ok=True)


class S3Storage(ObjectStorage):
    def __init__(self, bucket: str, region: str, endpoint_url: str = ""):
        import boto3  # local import: keep boto3 optional for local-only dev

        self.bucket = bucket
        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            region_name=region or None,
            endpoint_url=endpoint_url or None,
        )

    def put_file(self, local_path: str, object_key: str) -> None:
        self.client.upload_file(local_path, self.bucket, object_key)

    def get_local_path(self, object_key: str, download_to: str) -> str:
        os.makedirs(os.path.dirname(download_to) or ".", exist_ok=True)
        self.client.download_file(self.bucket, object_key, download_to)
        return download_to

    def exists(self, object_key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except ClientError:
            return False

    def delete(self, object_key: str) -> None:
        # S3's delete_object is idempotent -- a missing key is not an error,
        # matching the interface's "already gone is success" contract.
        self.client.delete_object(Bucket=self.bucket, Key=object_key)


def get_storage() -> ObjectStorage:
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 but S3_BUCKET is not set")
        return S3Storage(settings.s3_bucket, settings.s3_region, settings.s3_endpoint_url)
    return LocalFilesystemStorage(settings.storage_local_dir)


storage = get_storage()
