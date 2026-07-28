"""Ground/horizon masking for sky-clutter suppression.

Segments the sky region via a connected-component flood: the largest
edge-free component touching the top of the frame is treated as sky.
Interior holes within that sky region (e.g. a drone's silhouette or a
cloud pocket) are absorbed back into sky so the drone remains searchable.
Everything else is ground; the mask is 255 there and 0 in sky.

The mask is fed into ``MotionDetector`` as ``diff[mask > 0] = 0``, so
ground clutter does not generate false motion events while drones flying
in the sky region are fully visible.

NOTE: the companion ``MotionDetector`` lives in
``dronetracker/vision/motion_detector.py``.
"""

__all__ = [
    "GroundMaskGenerator",
    "MASK_DOWNSCALE",
]

import cv2
import numpy as np

MASK_DOWNSCALE = 2   # compute() runs at 1/MASK_DOWNSCALE; mask upscaled at end


class GroundMaskGenerator:
    """Compute and cache a ground/sky binary mask for a given video frame.

    Call ``compute(frame)`` whenever the scene changes (e.g. after a camera
    pan/tilt settles).  ``get_mask()`` returns the most recent mask.

    ``SHOW_IMAGE = False`` (default) — set to True only for debugging; setting
    it True on a headless machine will raise a cv2 error.
    """

    WINDOW_NAME   = "mask"
    SHOW_IMAGE    = False

    def __init__(self):
        self.mask        = None
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
        If no edge-free connected component touches the top row (e.g. camera
        tilted fully into ground), the previous mask is returned unchanged
        (None on the very first frame, which means no suppression).
        """
        h, w   = frame.shape[:2]
        sh, sw = h // MASK_DOWNSCALE, w // MASK_DOWNSCALE

        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        # Cloud detection — high brightness, low saturation.
        hsv    = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        clouds = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30)

        # Edge map (cloud regions zeroed out so clouds don't fragment sky).
        gray     = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges    = cv2.Canny(gray, 50, 150)
        edges[clouds] = 0
        smoothed = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8))

        # ------------------------------------------------------------------ #
        # Sky segmentation: flood from the top row using connected components.
        # ------------------------------------------------------------------ #

        # Passable = pixels with no edge (background texture, open sky).
        passable  = (smoothed == 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(passable)

        # Collect component IDs that touch the top row.
        top_ids = np.unique(labels[0, :])
        top_ids = top_ids[top_ids != 0]   # exclude background label 0

        if top_ids.size == 0:
            # Nothing passable on the top row — keep previous mask.
            return self.mask

        # Pick the largest among them as the canonical sky blob.
        counts = np.bincount(labels.ravel())
        sky_id = int(top_ids[np.argmax(counts[top_ids])])
        sky    = (labels == sky_id).astype(np.uint8)

        # Fill interior holes: non-sky regions that don't touch any image
        # border are fully enclosed by sky (drone silhouettes, cloud pockets,
        # etc.) and should also be searchable, not masked as ground.
        notsky       = (sky == 0).astype(np.uint8)
        _, nlabels   = cv2.connectedComponents(notsky)
        # Only the bottom edge anchors a region as real ground.
        # Left/right-edge islands that sky wraps around are absorbed into sky.
        border_ids = np.unique(nlabels[-1, :])
        border_ids = border_ids[border_ids != 0]
        # Regions touching the border are real ground; everything else is interior.
        ground       = np.isin(nlabels, border_ids).astype(np.uint8)
        small_mask   = ground * np.uint8(255)   # 255 = ground, 0 = sky + holes

        # Dilate ground slightly to avoid a thin unmasked strip right at the
        # horizon edge.  Reduce/remove this kernel if drones near the horizon
        # get clipped.
        small_mask = cv2.dilate(small_mask, np.ones((7, 7), dtype=np.uint8))
        self.mask  = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self._has_display:
            cv2.imshow(GroundMaskGenerator.WINDOW_NAME, self.mask)
            cv2.waitKey(1)

        return self.mask

    def get_mask(self):
        """Return the most recently computed mask, or None if never computed."""
        return self.mask
