"""FFTrix Recording Retention Policy — nightly auto-cleanup of old recordings."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

_24H = 24 * 60 * 60  # seconds


class RetentionManager:
    """Deletes recording files older than each camera's retention_days setting.

    Usage::

        mgr = RetentionManager(db=db, recordings_root=RECORDINGS_DIR)
        mgr.start()  # runs run_once() immediately then every 24 h
    """

    def __init__(self, db, recordings_root: Path):
        self.db = db
        self.recordings_root = Path(recordings_root)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the nightly retention background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="RetentionManager")
        self._thread.start()
        log.info("RetentionManager started.")

    def stop(self) -> None:
        """Signal the background thread to stop at next sleep cycle."""
        self._stop_event.set()

    def run_once(self) -> dict[str, int]:
        """Run one retention pass immediately.

        Returns a dict mapping cam_id -> number of files deleted.
        """
        cameras = self.db.get_cameras()
        results: dict[str, int] = {}
        now = time.time()

        for cam in cameras:
            cam_id = cam["id"]
            retention_days = cam.get("retention_days", 30)
            if retention_days == 0:
                continue  # 0 = keep forever

            cam_dir = self.recordings_root / cam_id
            if not cam_dir.exists():
                continue

            cutoff = now - (retention_days * 86400)
            deleted = 0
            for f in cam_dir.iterdir():
                if not f.is_file():
                    continue
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        deleted += 1
                        log.debug("Deleted old recording: %s", f)
                except OSError as exc:
                    log.warning("Could not delete %s: %s", f, exc)

            if deleted:
                log.info("RetentionManager: deleted %d file(s) for camera %s (>%d days)", deleted, cam_id, retention_days)
            results[cam_id] = deleted

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        self.run_once()
        while not self._stop_event.wait(timeout=_24H):
            self.run_once()
