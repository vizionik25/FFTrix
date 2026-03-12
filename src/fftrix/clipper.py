"""FFTrix Clip Exporter — extract MP4 clips from recorded footage."""

from __future__ import annotations

import os
import shutil
import threading
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


class ClipExporter:
    """Extracts a time-ranged clip from a camera's recording library.

    Clips are stitched from the individual segment files produced by
    ``WriteGear`` and saved as a new MP4 in an export directory.

    Usage::

        exporter = ClipExporter(recordings_root=RECORDINGS_DIR,
                                export_dir=Path('~/.fftrix/exports').expanduser())
        clip_path = exporter.export(cam_id='CAM1',
                                    start_ts=1700000000,
                                    end_ts=1700003600)
    """

    def __init__(self, recordings_root: Path, export_dir: Path | None = None):
        self.recordings_root = Path(recordings_root)
        self.export_dir = export_dir or self.recordings_root.parent / 'exports'
        self.export_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_segments(self, cam_id: str) -> list[dict]:
        """Return sorted list of recording segments for a camera.

        Each dict has: ``path``, ``mtime`` (float epoch), ``size`` (bytes).
        """
        cam_dir = self.recordings_root / cam_id
        if not cam_dir.exists():
            return []
        segments = []
        for f in sorted(cam_dir.glob('*.mp4')):
            segments.append({
                'path': str(f),
                'mtime': f.stat().st_mtime,
                'size': f.stat().st_size,
                'name': f.name,
            })
        return segments

    def export(
        self,
        cam_id: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        callback: Callable[[str | None], None] | None = None,
    ) -> str | None:
        """Export matching segments as a concatenated MP4.

        Filters segments whose ``mtime`` falls in ``[start_ts, end_ts]``.
        If no time range is given, all segments are included.

        Args:
            cam_id: Camera ID whose recordings to search.
            start_ts: Epoch start (inclusive). None = no lower bound.
            end_ts: Epoch end (inclusive). None = no upper bound.
            callback: Optional ``fn(clip_path_or_None)`` called when done.

        Returns:
            Path to the exported clip, or None if nothing was found.
        """
        segments = self.list_segments(cam_id)
        selected = [
            s for s in segments
            if (start_ts is None or s['mtime'] >= start_ts)
            and (end_ts is None or s['mtime'] <= end_ts)
        ]
        if not selected:
            log.warning("No segments found for %s in range [%s, %s]", cam_id, start_ts, end_ts)
            if callback:
                callback(None)
            return None

        # If only one segment, just copy it
        if len(selected) == 1:
            return self._copy_single(selected[0]['path'], cam_id, callback)

        # Multiple segments: try ffmpeg concat, fall back to copy of first
        return self._concat_ffmpeg(selected, cam_id, callback)

    def export_async(
        self,
        cam_id: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        callback: Callable[[str | None], None] | None = None,
    ) -> None:
        """Non-blocking version of :meth:`export`.  Runs in a daemon thread."""
        threading.Thread(
            target=self.export,
            args=(cam_id, start_ts, end_ts, callback),
            daemon=True,
        ).start()

    def delete_segment(self, segment_path: str) -> bool:
        """Delete a single recording segment file.

        Returns True if deleted, False on error.
        """
        try:
            Path(segment_path).unlink()
            return True
        except Exception as exc:
            log.error("Failed to delete segment %s: %s", segment_path, exc)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _output_path(self, cam_id: str) -> str:
        import time
        ts = int(time.time())
        return str(self.export_dir / f"{cam_id}_clip_{ts}.mp4")

    def _copy_single(self, src: str, cam_id: str, callback):
        dst = self._output_path(cam_id)
        try:
            shutil.copy2(src, dst)
            log.info("Clip exported (single): %s", dst)
            if callback:
                callback(dst)
            return dst
        except Exception as exc:
            log.error("Clip copy failed: %s", exc)
            if callback:
                callback(None)
            return None

    def _concat_ffmpeg(self, segments: list[dict], cam_id: str, callback):
        """Use ffmpeg concat demuxer to stitch segments."""
        import subprocess
        import tempfile

        concat_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        try:
            for s in segments:
                concat_list.write(f"file '{s['path']}'\n")
            concat_list.close()

            dst = self._output_path(cam_id)
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list.name,
                '-c', 'copy', dst,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                log.info("Clip exported (concat): %s", dst)
                if callback:
                    callback(dst)
                return dst
            else:
                log.warning("ffmpeg failed: %s — falling back to single copy",
                            result.stderr.decode()[:200])
                return self._copy_single(segments[0]['path'], cam_id, callback)
        except FileNotFoundError:
            # ffmpeg not installed — copy first segment
            log.warning("ffmpeg not found; copying first segment only")
            return self._copy_single(segments[0]['path'], cam_id, callback)
        except Exception as exc:
            log.error("Clip concat failed: %s", exc)
            if callback:
                callback(None)
            return None
        finally:
            try:
                os.unlink(concat_list.name)
            except Exception:
                pass
