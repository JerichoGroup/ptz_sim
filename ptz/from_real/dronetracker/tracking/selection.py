"""Best-track selection for the PTZ tracking pipeline."""

__all__ = ["pick_best_track"]

import numpy as np


def pick_best_track(tracked_objects, frame_w: float, frame_h: float):
    """Choose the track ID to lock onto from a list of Norfair TrackedObjects.

    Selection criterion: longest-lived track (highest ``hit_counter``),
    with centre-of-frame distance as a tie-breaker.

    Args:
        tracked_objects: Iterable of Norfair ``TrackedObject`` instances from
                         ``tracker.update()``.
        frame_w:         Frame width in pixels.
        frame_h:         Frame height in pixels.

    Returns:
        The integer Norfair track ID of the best candidate, or ``None`` if
        no confirmed tracks exist (i.e. all objects have ``last_detection=None``).
    """
    cx, cy    = frame_w / 2.0, frame_h / 2.0
    best_id   = None
    best_age  = -1
    best_dist = float("inf")

    for obj in tracked_objects:
        if obj.last_detection is None:
            continue
        age  = obj.hit_counter
        ex, ey = obj.estimate[0]
        dist   = float(np.hypot(ex - cx, ey - cy))
        if age > best_age or (age == best_age and dist < best_dist):
            best_age  = age
            best_dist = dist
            best_id   = obj.id

    return best_id
