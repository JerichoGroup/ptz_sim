"""Track and bounding-box overlay drawing for the live viewer."""

__all__ = ["draw_tracks_overlay", "draw_yolo_box"]

# Per-class bounding-box colours (BGR).  Fallback cyan for unknown labels.
_BBOX_COLORS: dict = {
    "drone":      (0,   255, 255),   # cyan
    "bird":       (0,   200,   0),   # green
    "airplane":   (0,   128, 255),   # orange
    "helicopter": (180,   0, 255),   # purple
}
_BBOX_DEFAULT_COLOR = (0, 255, 255)  # cyan fallback


def draw_tracks_overlay(disp, tracks, locked_id, sx: float, sy: float,
                         track_history, now=None) -> None:
    """Render Norfair track dots, ID labels, and motion trails onto ``disp``.

    Args:
        disp:          BGR display frame (modified in-place).
        tracks:        List of Norfair ``TrackedObject`` instances.
        locked_id:     Track ID to highlight in red (None = no highlight).
        sx:            Horizontal scale factor from camera to display pixels.
        sy:            Vertical scale factor from camera to display pixels.
        track_history: The module-level ``defaultdict(list)`` from
                       ``dronetracker.tracking.trails`` (or ``utils.tracking``).
                       Mutated in-place: new points appended, capped at 50.
        now:           Current wall-clock time (``time.time()``).  When provided,
                       calls ``note_seen(track_id, now)`` so ``prune_stale``
                       can expire entries that stopped being seen.  Pass None
                       (default) to skip expiry bookkeeping (offline pipelines).
    """
    import cv2
    import numpy as np

    if now is not None:
        from dronetracker.tracking.trails import note_seen as _note_seen
    else:
        _note_seen = None

    for obj in tracks:
        if obj.last_detection is None:
            continue
        track_id    = obj.id
        ex, ey      = obj.estimate[0]
        dx, dy      = int(ex * sx), int(ey * sy)
        is_locked   = (track_id == locked_id)
        dot_color   = (0, 0, 255)  if is_locked else (0, 255, 0)
        trail_color = (255, 0, 0)  if is_locked else (180, 180, 0)
        radius      = 8            if is_locked else 5

        cv2.circle(disp, (dx, dy), radius, dot_color, -1)
        cv2.putText(disp, f"ID {track_id}", (dx + 8, dy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2)

        trail = track_history[track_id]
        trail.append((dx, dy))
        if _note_seen is not None:
            _note_seen(track_id, now)
        if len(trail) > 50:
            trail.pop(0)
        if len(trail) > 1:
            cv2.polylines(disp, [np.array(trail, dtype=np.int32)],
                          False, trail_color, 2)


def draw_yolo_box(disp, yolo_box, sx: float, sy: float) -> None:
    """Render a YOLO bounding box and confidence label (cyan) onto ``disp``.

    Args:
        disp:     BGR display frame (modified in-place).
        yolo_box: ``(x1, y1, x2, y2, conf, label)`` in camera-native pixels.
        sx:       Horizontal scale factor from camera to display pixels.
        sy:       Vertical scale factor from camera to display pixels.
    """
    import cv2

    yx1, yy1, yx2, yy2, yconf, ylabel = yolo_box
    color = _BBOX_COLORS.get(ylabel, _BBOX_DEFAULT_COLOR)
    ydx1 = int(yx1 * sx); ydy1 = int(yy1 * sy)
    ydx2 = int(yx2 * sx); ydy2 = int(yy2 * sy)
    cv2.rectangle(disp, (ydx1, ydy1), (ydx2, ydy2), color, 2)
    ytxt = f"{ylabel} {yconf:.2f}"
    cv2.putText(disp, ytxt, (ydx1, max(ydy1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(disp, ytxt, (ydx1, max(ydy1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
