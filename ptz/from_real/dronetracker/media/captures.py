# -*- coding: utf-8 -*-
"""Stateless image-capture helpers for the live PTZ pipeline.

These functions only write frames/crops to disk via ``cv2.imwrite``.  They hold
no state and take no shared objects, so they are safe to call from either the
detect thread (classified crops) or the main thread (freeze/manual screenshots)
without any cross-thread coupling.  ``cv2`` is imported lazily.
"""

__all__ = ["save_classified_crops", "save_frame"]

import os
from datetime import datetime


def save_classified_crops(frame, dets, out_dir=".") -> int:
    """Save per-detection bbox crops and one annotated full frame.

    Returns the number of crops written.  When ``dets`` is empty, saves the full
    frame instead and returns 0.  Files are written under ``out_dir`` (default
    the current working directory, preserving the original behaviour); pass a
    chunk's ``crops/`` directory to route them onto the SSD.
    """
    import cv2
    out_dir = str(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not dets:
        fn = os.path.join(out_dir, f"capture_{ts}.png")
        cv2.imwrite(fn, frame)
        print(f"[TRACK] no detections — saved full frame {fn}")
        return 0
    count = 0
    for i, (x1, y1, x2, y2, conf, label) in enumerate(dets):
        if y2 <= y1 or x2 <= x1:
            continue
        crop = frame[y1:y2, x1:x2]
        safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        fn = os.path.join(out_dir, f"crop_{safe_label}_{conf:.2f}_{ts}_{i}.png")
        cv2.imwrite(fn, crop)
        print(f"[TRACK] saved {fn}")
        count += 1
    annotated = frame.copy()
    for x1, y1, x2, y2, conf, label in dets:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {conf:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    fn = os.path.join(out_dir, f"annotated_{ts}.png")
    cv2.imwrite(fn, annotated)
    print(f"[TRACK] saved annotated {fn}")
    return count


def save_frame(frame, prefix: str) -> str:
    """Write ``frame`` to a timestamped ``{prefix}_YYYYmmdd_HHMMSS.png`` file.

    Returns the filename written.  Does not print — the caller logs with its own
    label so the existing log lines stay unchanged.
    """
    import cv2
    fn = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S.png")
    cv2.imwrite(fn, frame)
    return fn
