"""Utility helpers for reading/writing files on Google Cloud Storage.

When running locally (no GCS_BUCKET env var set) every function is a no-op and
paths are passed through unchanged, so the rest of the codebase works identically
on a developer laptop and inside a Vertex AI training container.

Usage pattern:
    from gcs import resolve, upload_dir, upload_file

    # Download a GCS path to a local cache, return the local path:
    local = resolve("gs://bucket/data/sp500_history.csv")

    # Upload a local directory to GCS when training finishes:
    upload_dir("checkpoints/", "gs://bucket/checkpoints/")
"""

import os
import shutil
import tempfile
from pathlib import Path

# Set GCS_BUCKET=my-bucket (without gs://) in your container env vars.
_BUCKET = os.environ.get("GCS_BUCKET", "")
_LOCAL_CACHE = os.environ.get("GCS_LOCAL_CACHE", "/tmp/gcs_cache")


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def resolve(path: str) -> str:
    """Return a local file path, downloading from GCS if needed.

    If *path* is already a local path (or GCS is not configured) it is returned
    unchanged.  Otherwise the object is downloaded to a local cache directory and
    that path is returned so callers need no GCS-specific code.
    """
    if not _is_gcs(path):
        return path

    try:
        from google.cloud import storage  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-storage is required when using GCS paths. "
            "Install it with: pip install google-cloud-storage"
        ) from e

    # Strip gs://bucket/ prefix → blob name
    without_prefix = path[len("gs://"):]
    bucket_name, _, blob_name = without_prefix.partition("/")

    local_path = os.path.join(_LOCAL_CACHE, blob_name)
    if os.path.exists(local_path):
        return local_path  # already cached

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[gcs] Downloading {path} → {local_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path


def resolve_dir(gcs_prefix: str, local_dir: str) -> str:
    """Download all objects under *gcs_prefix* into *local_dir*.

    Returns *local_dir* so callers can use it directly as a directory path.
    If *gcs_prefix* is not a GCS path the function is a no-op.
    """
    if not _is_gcs(gcs_prefix):
        return gcs_prefix

    try:
        from google.cloud import storage  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-storage is required when using GCS paths."
        ) from e

    without_prefix = gcs_prefix[len("gs://"):]
    bucket_name, _, prefix = without_prefix.partition("/")

    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    print(f"[gcs] Downloading {len(blobs)} objects from {gcs_prefix} → {local_dir}")
    for blob in blobs:
        # Preserve sub-directory structure under local_dir
        relative = blob.name[len(prefix):].lstrip("/")
        local_path = os.path.join(local_dir, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

    return local_dir


def upload_file(local_path: str, gcs_dest: str) -> None:
    """Upload a single local file to a GCS destination path.

    No-op when *gcs_dest* is not a GCS path.
    """
    if not _is_gcs(gcs_dest):
        return

    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        print("[gcs] WARNING: google-cloud-storage not installed, skipping upload.")
        return

    without_prefix = gcs_dest[len("gs://"):]
    bucket_name, _, blob_name = without_prefix.partition("/")

    print(f"[gcs] Uploading {local_path} → {gcs_dest}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def upload_dir(local_dir: str, gcs_prefix: str) -> None:
    """Recursively upload *local_dir* to *gcs_prefix*.

    No-op when *gcs_prefix* is not a GCS path.
    """
    if not _is_gcs(gcs_prefix):
        return

    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        print("[gcs] WARNING: google-cloud-storage not installed, skipping upload.")
        return

    without_prefix = gcs_prefix[len("gs://"):]
    bucket_name, _, prefix = without_prefix.partition("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_dir = Path(local_dir)
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            blob_name = f"{prefix}/{relative}".replace("\\", "/")
            print(f"[gcs] Uploading {file_path} → gs://{bucket_name}/{blob_name}")
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
