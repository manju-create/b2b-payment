"""
RecoverFlow — persistent upload storage
=======================================
Writes debtor-uploaded documents to a persistent volume (or an S3-compatible
bucket) so MongoDB only ever stores metadata, never raw binary — raw bytes in
Mongo would bloat documents and slow down reads.

Point ``UPLOAD_DIR`` at the mounted Railway volume path; the default is a local
``storage/uploads/`` directory (gitignored). To swap in S3, replace the body of
``save_upload`` with a boto3 ``put_object`` and keep the same metadata shape.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def upload_dir() -> Path:
    """Return the upload directory, creating it if necessary."""
    d = Path(os.environ.get("UPLOAD_DIR", str(REPO_ROOT / "storage" / "uploads")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(invoice_id: str, file_name: str, content: bytes) -> dict:
    """Write a file to persistent storage and return its metadata.

    Returns the exact shape pushed to the Mongo ``documents`` array:
      {"file_name", "url", "uploaded_at"}
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    safe_name = re.sub(r"[^\w.\-]", "_", file_name or "document")
    stored = f"{invoice_id}_{ts}_{safe_name}"
    (upload_dir() / stored).write_bytes(content)
    return {
        "file_name": file_name,
        "url": f"/uploads/{stored}",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
