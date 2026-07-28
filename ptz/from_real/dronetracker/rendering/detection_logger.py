"""DetectionLogger: appends YOLO detections to a CSV file, one row per box.

CSV columns: frame_index, time_sec, x1, y1, x2, y2, conf, label
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class DetectionLogger:
    """Appends YOLO detections to a CSV file, one row per detected box.

    Each add_detection() call opens the file, appends one row, and closes it —
    no in-memory buffer, no periodic flush, no O(n²) full-rewrite.

    Usage:
        logger = DetectionLogger()
        for box in yolo_boxes:
            logger.add_detection(box, frame_index)
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
                f"__{now.hour}_{now.minute}_detections.csv"
            )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = out_dir / filename

        with open(self.filepath, "w", newline="") as f:
            csv.writer(f).writerow(
                ["frame_index", "time_sec", "x1", "y1", "x2", "y2", "conf", "label"]
            )

    def add_detection(
        self,
        yolo_box: Tuple[float, float, float, float, float, str],
        frame_index: int,
    ) -> None:
        """Append one detected box as a single CSV row."""
        x1, y1, x2, y2, conf, label = yolo_box
        elapsed = round(time.time() - self.start_time, 4)
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow([
                frame_index, elapsed,
                round(float(x1), 2), round(float(y1), 2),
                round(float(x2), 2), round(float(y2), 2),
                round(float(conf), 4), label,
            ])

    def flush(self) -> None:
        """No-op: every write is immediate; nothing to flush."""
