"""Ground/horizon masking for sky-clutter suppression.

Finds the horizon line via a dynamic-programming least-cost path across the
edge map (``find_horizon_path_fast`` is the optimised version;
``find_horizon_path`` is the slow O(h*w) reference), then builds a binary
mask that covers everything *below* the horizon.  The mask is fed into
``MotionDetector`` so ground clutter does not generate false motion events.

NOTE: ``MotionDetector`` is intentionally kept in ``utils/MotionDetector.py``
(not moved here) because its module-level ``MOVE_MAX_BLOB_AREA`` attribute is
mutated at runtime by the pipeline.  This module imports from and re-exports
the relevant parts of that module.
"""

__all__ = [
    "find_horizon_path_fast",
    "find_horizon_path",
    "create_mask_from_path",
    "GroundMaskGenerator",
    "DOWNSCALE",
    "MASK_DOWNSCALE",
]

import cv2
import numpy as np
from tqdm import tqdm  # noqa: F401  (imported by original, keep for callers that relied on it)

DOWNSCALE      = 6   # horizon DP runs at 1/DOWNSCALE; path upscaled to full width
MASK_DOWNSCALE = 2   # compute() runs at 1/MASK_DOWNSCALE; mask upscaled at end


def find_horizon_path_fast(smooth, center_row=None):
    """Optimised DP least-cost horizon path.

    Args:
        smooth:     Binary/float edge image (bright pixels = edges).
        center_row: Full-resolution row to anchor the path around
                    (from the density-contrast estimate in ``compute()``).
                    Cost rises with distance from this row so cloud edges
                    far above can't outbid the true horizon.
                    If None, falls back to the original top-preference cost
                    (legacy / unit-test behaviour).

    Returns:
        1-D int32 array of length ``frame_width`` giving the horizon row at
        each column in full-resolution coordinates.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    h, w = smooth.shape
    ds   = DOWNSCALE if min(h, w) >= 512 else 1

    if ds == 1:
        # Small input (e.g. unit-test frames): run at full resolution using the
        # original algorithm so test thresholds are unaffected.
        small = smooth
        hs, ws = h, w
    else:
        small = cv2.resize(smooth, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        hs, ws = small.shape

    DY   = 4
    cost = np.ones((hs, ws), dtype=np.float32)
    if center_row is not None:
        center_small = int(center_row * hs / h)
        row_dist = np.abs(np.arange(hs, dtype=np.float32) - center_small)[:, None] / hs
        cost += 2.0 * row_dist
    else:
        cost += np.arange(hs, dtype=np.float32)[:, None] / hs
    cost -= 2.0 * (small > 0)

    dp    = np.empty((hs, ws), dtype=np.float32)
    dp[:, 0] = cost[:, 0]

    for x in range(1, ws):
        padded = np.full(hs + 2 * DY, np.inf, dtype=np.float32)
        padded[DY : DY + hs] = dp[:, x - 1]
        best      = sliding_window_view(padded, 2 * DY + 1).min(axis=1)
        dp[:, x]  = cost[:, x] + best

    # Backtrack: reconstruct path on the fly (no stored parent matrix needed).
    y          = int(np.argmin(dp[:, -1]))
    path_small = np.empty(ws, dtype=np.int32)
    for x in range(ws - 1, -1, -1):
        path_small[x] = y
        if x > 0:
            lo = max(0, y - DY)
            hi = min(hs, y + DY + 1)
            y  = lo + int(np.argmin(dp[lo:hi, x - 1]))

    if ds == 1:
        return path_small

    # Upscale path from low-res to full-res with linear interpolation.
    xs   = np.arange(ws) * (w / ws)
    path = np.interp(np.arange(w), xs, path_small * (h / hs))
    return np.clip(np.rint(path), 0, h - 1).astype(np.int32)


def find_horizon_path(smooth):
    """Slow O(h×w) reference DP implementation.

    Prefer ``find_horizon_path_fast`` for production use.
    """
    h, w = smooth.shape
    cost   = np.ones((h, w), dtype=np.float32)
    cost  -= np.arange(h, dtype=np.float32).reshape(-1, 1) / h   # prefer lower rows
    cost  -= 2 * (smooth > 0).astype(np.float32)

    dp     = np.full((h, w), np.inf, dtype=np.float32)
    parent = np.full((h, w), -1,    dtype=np.int32)
    dp[:, 0] = cost[:, 0]

    for x in range(1, w):
        for y in range(h):
            best_prev, best_cost = y, dp[y, x - 1]
            for dy in [-4, -3, -2, -1, 1, 2, 3, 4]:
                py = y + dy
                if 0 <= py < h and dp[py, x - 1] < best_cost:
                    best_cost = dp[py, x - 1]
                    best_prev = py
            dp[y, x]     = cost[y, x] + best_cost
            parent[y, x] = best_prev

    y    = int(np.argmin(dp[:, -1]))
    path = np.zeros(w, dtype=np.int32)
    for x in range(w - 1, -1, -1):
        path[x] = y
        y        = parent[y, x]
    return path


def create_mask_from_path(path, shape):
    """Build a binary mask where all pixels *below* ``path`` are 255.

    Args:
        path:  1-D int array of horizon row per column.
        shape: (height, width) of the desired mask.

    Returns:
        uint8 numpy array of shape ``(height, width)``.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        mask[path[x] :, x] = 255
    return cv2.dilate(mask, np.ones((5, 5)))


class GroundMaskGenerator:
    """Compute and cache a ground/sky binary mask for a given video frame.

    Call ``compute(frame)`` whenever the scene changes (e.g. after a camera
    pan/tilt settles).  ``get_mask()`` returns the most recent mask.

    ``SHOW_IMAGE = False`` (default) — set to True only for debugging; setting
    it True on a headless machine will raise a cv2 error.
    """

    WINDOW_NAME   = "mask"
    SHOW_IMAGE    = False
    WARP_DEBUG_PATH = "/tmp/gmg_warp_debug.png"

    def __init__(self):
        self.mask        = None
        self.anchor_mask = None
        self._has_display = False
        if self.SHOW_IMAGE:
            try:
                cv2.namedWindow(GroundMaskGenerator.WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(
                    GroundMaskGenerator.WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
                self._has_display = True
            except cv2.error:
                pass

    def compute(self, frame):
        """Recompute the ground mask from ``frame``.

        Returns the new mask (also stored as ``self.mask``).
        If the scene has insufficient contrast to find a reliable horizon, the
        previous mask is returned unchanged.
        """
        h, w   = frame.shape[:2]
        sh, sw = h // MASK_DOWNSCALE, w // MASK_DOWNSCALE

        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        # Cloud detection — high brightness, low saturation.
        hsv    = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        clouds = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30)

        # Edge map (cloud regions zeroed out).
        gray     = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges    = cv2.Canny(gray, 50, 150)
        edges[clouds] = 0
        smoothed = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8))

        # Density-contrast horizon estimate.
        density       = (smoothed > 0).sum(axis=1).astype(np.float64)
        cumsum        = np.cumsum(density)
        n             = np.arange(1, sh + 1, dtype=np.float64)
        density_above = cumsum / n
        density_below = (cumsum[-1] - cumsum) / np.maximum(sh - n, 1)
        dc_score      = density_below - density_above
        dc_horizon    = int(np.argmax(dc_score))
        dc_contrast   = float(dc_score.max())

        # Insufficient contrast — return previous mask if available.
        if dc_contrast < 100.0 / MASK_DOWNSCALE and self.mask is not None:
            return self.mask

        # DP horizon path anchored to the density-contrast estimate.
        path        = find_horizon_path_fast(smoothed, center_row=dc_horizon)
        above_path  = np.arange(sh)[:, None] < path[None, :]

        # Sky flood-fill from the top-right corner.
        passable        = ((smoothed == 0) & above_path).astype(np.uint8)
        _, labels       = cv2.connectedComponents(passable)
        top_labels      = set(labels[0, sw // 2:].tolist()) - {0}

        if top_labels:
            lut           = np.zeros(labels.max() + 1, dtype=np.uint8)
            lut[list(top_labels)] = 255
            small_mask    = 255 - lut[labels]
        else:
            small_mask    = create_mask_from_path(path, (sh, sw))

        # Island removal — keep only ground-connected regions.
        _, island_labels = cv2.connectedComponents((small_mask == 255).astype(np.uint8))
        ground_labels    = np.unique(island_labels[~above_path])
        ground_labels    = ground_labels[ground_labels != 0]
        if len(ground_labels):
            lut              = np.zeros(island_labels.max() + 1, dtype=np.uint8)
            lut[ground_labels] = 255
            small_mask       = lut[island_labels]

        small_mask = cv2.dilate(small_mask, np.ones(7))
        self.mask  = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        self.anchor_mask = self.mask.copy()

        if self._has_display:
            cv2.imshow(GroundMaskGenerator.WINDOW_NAME, self.mask)
            cv2.waitKey(1)

        return self.mask

    def get_mask(self):
        """Return the most recently computed mask, or None if never computed."""
        return self.mask
