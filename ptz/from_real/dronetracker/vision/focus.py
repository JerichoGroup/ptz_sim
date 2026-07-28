# -*- coding: utf-8 -*-
"""Focus-sharpness helpers for autofocus gating.

Pure cv2/numpy helpers used by the live PTZ TRACK state machine to decide when
the lens has converged: take a center crop of the frame and measure its
high-frequency content via Laplacian variance.  ``cv2`` is imported lazily so
this module stays cheap to import on machines without OpenCV.
"""

__all__ = ["center_gray_roi", "roi_sharpness"]

import numpy as np


def center_gray_roi(frame: np.ndarray, roi_frac: float) -> np.ndarray:
    """Return a grayscale center crop — (roi_frac × min_dim) square."""
    import cv2
    fh, fw = frame.shape[:2]
    half = max(1, int(min(fh, fw) * roi_frac / 2))
    cy, cx = fh // 2, fw // 2
    roi = frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    if frame.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi


def roi_sharpness(gray_roi: np.ndarray) -> float:
    """Laplacian variance of a grayscale ROI — cheap focus-sharpness metric."""
    import cv2
    if gray_roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
