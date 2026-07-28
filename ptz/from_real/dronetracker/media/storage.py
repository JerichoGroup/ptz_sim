# -*- coding: utf-8 -*-
"""Segmented, size-capped on-disk storage for the live PTZ pipeline.

``StorageManager`` routes everything the live pipeline saves — the recorded
video, the detection/track JSON logs, and the classified-crop PNGs — into one
folder *per recording chunk* under a configurable base directory (typically an
SSD mount).  Each chunk folder is named by its start time so the layout is::

    <base_dir>/
      20260628_143000/
        video.avi
        detections.json
        tracks.json
        crops/<crop PNGs>
      20260628_143500/
        ...

A new chunk is started every ``segment_seconds``.  After each rotation the
combined on-disk size of all chunk folders is enforced against ``max_size_gb``:
the **oldest whole chunk folders** are deleted (never the current one) until the
total fits.  Deleting a whole folder removes a chunk's video together with its
JSON and crops, so related data always comes and goes as a unit — you never end
up with detections whose video has been trimmed away.

Threading: the display thread is the single rotation/quota authority (it drives
the video-writer cadence); the detect thread only *appends* JSON/crops to the
current chunk.  All public methods take an internal lock so a rotation can swap
the per-chunk loggers without corrupting an in-flight append.
"""

__all__ = ["StorageManager"]

import os
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dronetracker.config.schema import StorageConfig
from dronetracker.rendering.detection_logger import DetectionLogger
from dronetracker.rendering.track_logger import TrackLogger

# Chunk folder name: YYYYmmdd_HHMMSS, with an optional _N suffix to disambiguate
# two chunks that start within the same second (sorts after the un-suffixed one).
_SEGMENT_RE = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")
_SEGMENT_FMT = "%Y%m%d_%H%M%S"
_GIB = 1024 ** 3



class StorageManager:
    """Owns the current recording chunk and enforces the rolling size cap."""

    def __init__(self, cfg: StorageConfig):
        self._cfg = cfg
        self._base = Path(cfg.base_dir)
        self._max_bytes = int(cfg.max_size_gb * _GIB) if cfg.max_size_gb > 0 else 0
        self._lock = threading.Lock()

        self._segment_dir: Optional[Path] = None
        self._segment_start: float = 0.0
        self._det_logger: Optional[DetectionLogger] = None
        self._trk_logger: Optional[TrackLogger] = None

    # ── Segment lifecycle ─────────────────────────────────────────────────────

    def start_segment(self) -> Path:
        """Finalise the current chunk and begin a fresh one; return its folder.

        Creates ``<base_dir>/<timestamp>/`` and fresh detection/track loggers
        inside it.  Called by the display thread at startup and on each rotation.
        """
        with self._lock:
            # Persist the tail of the chunk we are leaving before dropping it.
            self._flush_loggers_locked()

            seg = self._unique_segment_dir()
            seg.mkdir(parents=True, exist_ok=True)
            self._segment_dir = seg
            self._segment_start = time.time()
            self._det_logger = (
                DetectionLogger(str(seg), "detections.csv")
                if self._cfg.save_detections else None
            )
            self._trk_logger = (
                TrackLogger(str(seg), "tracks.csv")
                if self._cfg.save_tracks else None
            )
            print(f"[Storage] new chunk → {seg}", flush=True)
            return seg

    def close(self) -> None:
        """Flush the current chunk's JSON logs (call on shutdown)."""
        with self._lock:
            self._flush_loggers_locked()

    def should_rotate(self, now: float) -> bool:
        """True if the current chunk has run for at least ``segment_seconds``."""
        with self._lock:
            if self._segment_dir is None:
                return False
            return (now - self._segment_start) >= self._cfg.segment_seconds

    def video_path(self) -> Optional[Path]:
        """Path to the current chunk's ``video.avi`` (None if no chunk started)."""
        with self._lock:
            if self._segment_dir is None:
                return None
            return self._segment_dir / "video.avi"

    def crops_dir(self) -> Optional[Path]:
        """Current chunk's ``crops/`` dir (lazily created); None if disabled/no chunk."""
        with self._lock:
            if self._segment_dir is None or not self._cfg.save_crops:
                return None
            d = self._segment_dir / "crops"
            d.mkdir(parents=True, exist_ok=True)
            return d

    # ── Per-frame logging (called from the detect thread) ──────────────────────

    def add_detection(self, box, frame_index: int) -> None:
        """Append one YOLO box (x1,y1,x2,y2,conf,label) to the chunk's JSON."""
        with self._lock:
            if self._det_logger is not None:
                self._det_logger.add_detection(box, frame_index)

    def add_track_frame(self, tracks, frame_index: int, locked_id=None) -> None:
        """Append one frame's Norfair tracks to the chunk's JSON."""
        with self._lock:
            if self._trk_logger is not None:
                self._trk_logger.add_frame(tracks, frame_index, locked_id)

    # ── Quota enforcement ──────────────────────────────────────────────────────

    def enforce_quota(self) -> None:
        """Delete oldest whole chunk folders until total size <= max_size_gb.

        No-op when the quota is unlimited (``max_size_gb`` == 0).  Never deletes
        the current chunk.  Safe to run without holding the lock for the deletion
        itself: other threads only ever write to the current chunk, which is
        excluded from the candidate list.
        """
        if self._max_bytes <= 0:
            return

        with self._lock:
            current = self._segment_dir.resolve() if self._segment_dir else None

        chunks = self._list_chunk_dirs()
        total = sum(self._dir_size(d) for d in chunks)
        if total <= self._max_bytes:
            return

        # Oldest first.  Sort by parsed (timestamp, numeric-suffix) rather than
        # the raw name so collision suffixes order numerically ("_2" before
        # "_10"), not lexicographically.
        for d in sorted(chunks, key=self._chunk_sort_key):
            if total <= self._max_bytes:
                break
            if current is not None and d.resolve() == current:
                continue  # never delete the chunk currently being written
            size = self._dir_size(d)
            try:
                shutil.rmtree(d)
            except OSError as e:
                print(f"[Storage] could not delete {d}: {e}", flush=True)
                continue
            total -= size
            print(f"[Storage] over quota — deleted oldest chunk {d.name} "
                  f"({size / _GIB:.2f} GiB)", flush=True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _flush_loggers_locked(self) -> None:
        """Persist any buffered JSON for the outgoing chunk. Caller holds lock."""
        if self._det_logger is not None:
            self._det_logger.flush()
        if self._trk_logger is not None:
            self._trk_logger.flush()

    def _unique_segment_dir(self) -> Path:
        """Return a non-colliding ``<base>/<timestamp>`` path (caller holds lock)."""
        stamp = datetime.now().strftime(_SEGMENT_FMT)
        seg = self._base / stamp
        n = 2
        while seg.exists():
            seg = self._base / f"{stamp}_{n}"
            n += 1
        return seg

    @staticmethod
    def _chunk_sort_key(path: Path):
        """Sort key giving true chronological order for chunk folder names.

        ``<timestamp>`` sorts before ``<timestamp>_2`` before ``<timestamp>_10``.
        The un-suffixed folder is created first, so it ranks as suffix 1 (no
        ``_1`` is ever generated — suffixes start at _2).
        """
        m = re.match(r"^(\d{8}_\d{6})(?:_(\d+))?$", path.name)
        if not m:
            return (path.name, 0)
        return (m.group(1), int(m.group(2)) if m.group(2) else 1)

    def _list_chunk_dirs(self) -> list:
        """All immediate sub-folders of base_dir matching the chunk-name pattern."""
        if not self._base.exists():
            return []
        out = []
        for entry in os.scandir(self._base):
            if entry.is_dir() and _SEGMENT_RE.match(entry.name):
                out.append(Path(entry.path))
        return out

    @staticmethod
    def _dir_size(path: Path) -> int:
        """Total size (bytes) of all files under ``path``; missing files skipped."""
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass  # file vanished mid-scan — ignore
        return total
