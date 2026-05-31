"""Snapshots the events SQLite DB to a private Hugging Face Dataset repo so
the data survives Space restarts and feeds the retraining pipeline.

Requires the `HF_WRITE_TOKEN` Space secret (set in HF Spaces UI), with write
access to the dataset repo `DATASET_REPO`. Falls back to a no-op if missing.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi


DATASET_REPO = os.environ.get("FITPHONE_EVENTS_DATASET", "zekerheydArthur/fitphone-events")
# Minimum seconds between uploads — avoids hammering the Hub on every event.
SNAPSHOT_INTERVAL_S = int(os.environ.get("FITPHONE_SNAPSHOT_INTERVAL", "300"))


class EventSnapshotter:
    """Async-friendly snapshotter. Call [maybe_snapshot] from request handlers;
    it returns immediately and uploads on a background thread when due."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._last_upload = 0.0
        self._lock = threading.Lock()
        self._in_flight = False
        token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
        self._api = HfApi(token=token) if token else None

    def maybe_snapshot(self) -> None:
        """Fire-and-forget. No-op if a recent upload happened or no token."""
        if self._api is None:
            return
        now = time.time()
        with self._lock:
            if self._in_flight:
                return
            if now - self._last_upload < SNAPSHOT_INTERVAL_S:
                return
            self._in_flight = True
            self._last_upload = now
        threading.Thread(target=self._upload, daemon=True).start()

    def _upload(self) -> None:
        try:
            if not self.db_path.exists():
                return
            self._api.upload_file(
                path_or_fileobj=str(self.db_path),
                path_in_repo="events.db",
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message=f"snapshot {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            )
            print(f"[persistence] uploaded {self.db_path} to {DATASET_REPO}")
        except Exception as e:
            print(f"[persistence] upload failed: {e}")
        finally:
            with self._lock:
                self._in_flight = False
