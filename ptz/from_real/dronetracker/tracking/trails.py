"""Norfair track-history, detection helpers, and box-based drawing.

``track_history`` is a module-level singleton (a plain ``defaultdict``) shared
by all scripts that import it.  It accumulates per-track centroid trails keyed
by track ID.

Trail expiry helpers (``note_seen``, ``prune_stale``, ``clear_trails``) let the
live pipeline delete stale trails so that track IDs that disappear (or get
recycled after a tracker reset) do not produce spurious jump-lines.

The ``draw_tracks`` function renders bounding boxes and trails for YOLO-based
(box) detections.  For centroid-only pipelines see ``rendering/draw.py``.
"""

__all__ = [
    "track_history",
    "note_seen",
    "prune_stale",
    "clear_trails",
    "create_norfair_detections",
    "draw_tracks",
]

import cv2
import numpy as np
from collections import defaultdict
from norfair import Detection


# Shared trail history: {track_id: [(cx, cy), ...]} capped at 50 entries.
# Module-level singleton — importers get the same dict object.
track_history: defaultdict = defaultdict(list)

# Last-seen wall-clock time for each track id.  Parallel to track_history.
# Updated by note_seen(); pruned by prune_stale(); wiped by clear_trails().
track_last_seen: dict = {}


def note_seen(track_id: int, now: float) -> None:
    """Record that ``track_id`` was seen at wall-clock time ``now``.

    Called from drawing code each time a track point is appended so
    ``prune_stale`` can expire entries that have not been updated recently.
    """
    track_last_seen[track_id] = now


def prune_stale(now: float, max_age_s: float) -> None:
    """Delete trail history for track IDs not seen in the last ``max_age_s`` seconds.

    Mutates ``track_history`` and ``track_last_seen`` in place so the singleton
    object identities are preserved (required — see CLAUDE.md).
    """
    stale = [tid for tid, t in track_last_seen.items() if now - t > max_age_s]
    for tid in stale:
        track_history.pop(tid, None)
        del track_last_seen[tid]


def clear_trails() -> None:
    """Wipe all trail history and last-seen timestamps in place.

    Call this on every tracker reset (``_reset_to_idle``) so that recycled
    Norfair IDs do not inherit points from a previous session.
    """
    track_history.clear()
    track_last_seen.clear()


def create_norfair_detections(boxes, scores) -> list:
    """Convert [x1,y1,x2,y2] boxes + scores to Norfair Detection objects.

    The box is stashed as ``detection.box`` so drawing code can access it.

    Args:
        boxes:  List of [x1, y1, x2, y2] bounding boxes.
        scores: Confidence scores (one per box).

    Returns:
        List of ``norfair.Detection`` objects.
    """
    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        det = Detection(
            points=np.array([[cx, cy]]),
            scores=np.array([score]),
        )
        det.box = box
        detections.append(det)
    return detections


def draw_tracks(frame, tracked_objects):
    """Draw bounding boxes, IDs, and centroid trails for tracked objects.

    Uses ``obj.last_detection.box`` so this is only suitable for box-based
    detections (e.g. from ``create_norfair_detections``).

    Args:
        frame:           BGR image to draw onto (modified in-place).
        tracked_objects: List of Norfair TrackedObject instances.

    Returns:
        The modified ``frame``.
    """
    for obj in tracked_objects:
        if obj.last_detection is None:
            continue

        track_id = obj.id
        x1, y1, x2, y2 = map(int, obj.last_detection.box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        trail  = track_history[track_id]
        trail.append(center)
        if len(trail) > 50:
            trail.pop(0)

        if len(trail) > 1:
            cv2.polylines(
                frame,
                [np.array(trail, dtype=np.int32)],
                False,
                (255, 0, 0),
                2,
            )

    return frame
