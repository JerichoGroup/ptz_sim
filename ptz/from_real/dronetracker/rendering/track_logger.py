"""TrackLogger: appends Norfair track positions to a CSV file.

CSV columns: frame_index, time_sec, track_id, x, y, is_locked
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class TrackLogger:
    """Appends Norfair track positions to a CSV file, one row per track per frame.

    Each add_frame() call opens the file, appends one row per active track, and
    closes it — no in-memory buffer, no periodic flush, no O(n²) full-rewrite.

    Usage:
        logger = TrackLogger()
        logger.add_frame(tracks, frame_index, locked_id)
    """

    def __init__(
        self,
        output_dir: str = ".",
        filename: Optional[str] = None,
    ) -> None:
        self.start_time = time.time()

        if filename is None:
            now = datetime.now()
            filename = (
                f"{now.day}_{now.month}_{now.year}"
                f"__{now.hour}_{now.minute}_tracks.csv"
            )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = out_dir / filename

        with open(self.filepath, "w", newline="") as f:
            csv.writer(f).writerow(
                ["frame_index", "time_sec", "track_id", "x", "y", "is_locked"]
            )

    def add_frame(self, tracks, frame_index: int, locked_id=None) -> None:
        """Append one row per active track for this frame."""
        elapsed = round(time.time() - self.start_time, 4)
        rows = []
        for obj in tracks:
            if obj.last_detection is None:
                continue
            ex, ey = obj.estimate[0]
            rows.append([
                frame_index, elapsed, obj.id,
                round(float(ex), 4), round(float(ey), 4),
                obj.id == locked_id,
            ])
        if rows:
            with open(self.filepath, "a", newline="") as f:
                csv.writer(f).writerows(rows)

    def flush(self) -> None:
        """No-op: every write is immediate; nothing to flush."""
