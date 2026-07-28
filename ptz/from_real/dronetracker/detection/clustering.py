"""Blob extraction, clustering, and Norfair detection emission.

Split into three layers so the geometry can be tested without cv2 or norfair:

1. ``cluster_blobs``       — pure numpy, no imports beyond numpy.
2. ``extract_blobs``       — thin cv2 adapter (contour-find + bounding-rect).
3. ``motion_to_detections``— thin norfair adapter that calls 1+2 and emits
                             ``norfair.Detection`` objects with ``.area`` and
                             ``.box`` stashed for downstream auto-zoom logic.

The historical motion scripts each had their own copy of ``motion_to_detections``
with slightly different behaviour.  This unified version supersedes all three by
parameterising ``min_area`` and ``max_area``::

    # offline motion / batch (no upper bound):
    dets = motion_to_detections(mask, min_area=2, max_area=float("inf"))

    # live PTZ (also stashes .area / .box for auto-zoom):
    dets = motion_to_detections(mask, min_area=2, max_area=float("inf"))
"""

__all__ = ["cluster_blobs", "extract_blobs", "motion_to_detections"]

import numpy as np


# ---------------------------------------------------------------------------
# Pure geometry
# ---------------------------------------------------------------------------

def cluster_blobs(
    centroids: np.ndarray,
    areas: np.ndarray,
    cluster_distance: float,
) -> list:
    """Greedily cluster nearby blobs and keep the largest per cluster.

    Args:
        centroids:        (N, 2) float array of blob centres [cx, cy].
        areas:            (N,) float array of blob areas (same order).
        cluster_distance: Maximum centroid–centroid distance to merge two blobs.

    Returns:
        List of indices (into ``centroids``/``areas``) of the surviving blob
        from each cluster — one index per cluster, chosen by maximum area.
    """
    n    = len(centroids)
    used = [False] * n
    kept = []

    for i in range(n):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, n):
            if not used[j]:
                if np.linalg.norm(centroids[i] - centroids[j]) <= cluster_distance:
                    cluster.append(j)
                    used[j] = True
        best = max(cluster, key=lambda idx: areas[idx])
        kept.append(best)

    return kept


# ---------------------------------------------------------------------------
# cv2 adapter
# ---------------------------------------------------------------------------

def extract_blobs(
    motion_mask,
    min_area: float,
    max_area: float,
) -> tuple:
    """Find contours in ``motion_mask`` and return filtered blob properties.

    Args:
        motion_mask: uint8 binary mask (255 = motion).
        min_area:    Minimum contour area to keep (inclusive).
        max_area:    Maximum contour area to keep (inclusive).

    Returns:
        (centroids, areas, boxes) where each is a list of the same length N:
            centroids — list of np.array([cx, cy])
            areas     — list of float contour areas
            boxes     — list of (x1, y1, x2, y2) integer bounding rects
    """
    import cv2  # local import so the module-level is numpy-only

    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    centroids, areas, boxes = [], [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        centroids.append(np.array([x + w / 2, y + h / 2]))
        areas.append(area)
        boxes.append((x, y, x + w, y + h))
    return centroids, areas, boxes


# ---------------------------------------------------------------------------
# Norfair adapter
# ---------------------------------------------------------------------------

def motion_to_detections(
    motion_mask,
    min_area:         float = 2.0,
    max_area:         float = float("inf"),
    cluster_distance: float = 80.0,
    frame=None,
    appearance_fn=None,
    contrast_fn=None,
) -> list:
    """Convert a motion mask into a list of Norfair Detection objects.

    Finds contours, filters by area, clusters nearby blobs (keeping the
    largest per cluster), and emits one Detection per cluster at the blob
    centroid.  The chosen blob's bounding box and area are stashed on the
    Detection for downstream auto-zoom:

        det.area  → float contour area
        det.box   → (x1, y1, x2, y2) integer bounding rect

    When both ``frame`` and ``appearance_fn`` are supplied, an appearance
    descriptor is also stashed for collision-robust data association:

        det.embedding → appearance_fn(frame, det.box)  (may be None)

    Leaving either ``None`` keeps the legacy behaviour exactly (no embedding
    attribute set), so callers that don't use appearance — and the golden
    characterisation test — are unaffected.

    Args:
        motion_mask:      uint8 binary mask (255 = motion).
        min_area:         Minimum blob area to consider.
        max_area:         Maximum blob area to consider (use inf for no cap).
        cluster_distance: Maximum centroid–centroid distance to merge blobs.
        frame:            Optional source image (BGR or single-channel) the
                          mask was computed from, for appearance extraction.
        appearance_fn:    Optional ``(frame, box, motion_mask) -> descriptor | None``
                          used to compute ``det.embedding`` (e.g. a closure over
                          ``dronetracker.tracking.appearance.compute_descriptor``).
                          Receives the same ``motion_mask`` so the descriptor can
                          be restricted to the object's motion pixels.
        contrast_fn:      Optional ``(frame, box, motion_mask) -> float | None``
                          stashed as ``det.contrast`` (e.g. a closure over
                          ``dronetracker.tracking.appearance.blob_contrast``) for
                          sky/drift-track rejection by the track filter.

    Returns:
        List of ``norfair.Detection`` objects (empty if no blobs pass filters).
    """
    from norfair import Detection  # local import — keeps module top-level pure

    centroids, areas, boxes = extract_blobs(motion_mask, min_area, max_area)
    if not centroids:
        return []

    centroids_arr = np.array(centroids)
    areas_arr     = np.array(areas)

    kept = cluster_blobs(centroids_arr, areas_arr, cluster_distance)

    use_appearance = frame is not None and appearance_fn is not None
    use_contrast   = frame is not None and contrast_fn is not None

    detections = []
    for idx in kept:
        det      = Detection(points=np.array([centroids[idx]]))
        det.area = areas[idx]
        det.box  = boxes[idx]
        if use_appearance:
            # Pass the motion mask so the descriptor can be built over only the
            # object's motion pixels (excludes background inside the box).
            det.embedding = appearance_fn(frame, boxes[idx], motion_mask)
        if use_contrast:
            # Blob-vs-local-sky contrast — lets the track filter drop sky-like
            # "drift" detections (near-zero contrast) vs real blobs (high).
            det.contrast = contrast_fn(frame, boxes[idx], motion_mask)
        detections.append(det)
    return detections
