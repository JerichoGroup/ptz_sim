"""Track and bounding-box overlay drawing for the live viewer."""

__all__ = ["draw_tracks_overlay", "draw_yolo_box", "draw_aim_crosshair",
           "color_for_class"]

_FOLLOW_COLOR = (0,   0, 255)    # red  — actively-followed target (crosshair only)

# Visually-distinct BGR palette for per-class box colors.  Each detected class
# label is assigned the next palette entry on first sighting (see
# ``color_for_class``), so a given class keeps a stable, distinct color for the
# life of the process.  Red is deliberately omitted — it's reserved for the aim
# crosshair.
_CLASS_PALETTE = (
    (0, 255,   0),    # green
    (255, 255, 0),    # cyan
    (255, 128, 0),    # azure
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (0, 165, 255),    # orange
    (255, 0, 128),    # violet
    (128, 255, 0),    # spring green
    (255, 255, 128),  # pale cyan
    (128, 0, 255),    # pink-red
)
# label -> BGR, populated in first-seen order by ``color_for_class``.
_class_colors: dict = {}


def color_for_class(label) -> tuple:
    """Return a stable, visually-distinct BGR color for a class ``label``.

    Colors are handed out from ``_CLASS_PALETTE`` in first-seen order and cached,
    so the same class label always maps to the same color within a run and
    different classes get different colors (cycling the palette if there are more
    classes than palette entries).
    """
    key = str(label)
    color = _class_colors.get(key)
    if color is None:
        color = _CLASS_PALETTE[len(_class_colors) % len(_CLASS_PALETTE)]
        _class_colors[key] = color
    return color


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
        would_zoom  = getattr(obj, "auto_engage", False)
        if is_locked:
            dot_color, radius = (0, 0, 255), 8           # red = currently locked
        elif would_zoom:
            dot_color, radius = (255, 0, 255), 8         # magenta = auto-hunt would zoom
        else:
            dot_color, radius = (0, 255, 0), 5
        if is_locked:
            trail_color = (255, 0, 0)        # blue trail for the locked target
        elif would_zoom:
            trail_color = (255, 0, 255)      # magenta trail for the would-zoom track
        else:
            trail_color = (180, 180, 0)

        cv2.circle(disp, (dx, dy), radius, dot_color, -1)
        score = getattr(obj, "drone_score", None)
        label = f"ID {track_id}" if score is None else f"ID {track_id} {score:.2f}"
        if would_zoom and not is_locked:
            label = "ZOOM " + label
        cv2.putText(disp, label, (dx + 8, dy - 8),
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


def draw_yolo_box(disp, yolo_box, sx: float, sy: float, followed: bool = False) -> None:
    """Render the single TRACK-target box and label onto ``disp``.

    Args:
        disp:     BGR display frame (modified in-place).
        yolo_box: ``(x1, y1, x2, y2, conf, label)`` in camera-native pixels.
        sx:       Horizontal scale factor from camera to display pixels.
        sy:       Vertical scale factor from camera to display pixels.
        followed: True → thick box prefixed ``FOLLOW`` (the actively-followed
                  target the camera is chasing).  False → thinner box prefixed
                  ``YOLO`` (a raw / best-confidence fallback detection).  Box
                  color is per-class (see ``color_for_class``) in both cases;
                  followed vs raw is distinguished by border thickness + prefix.
    """
    import cv2

    yx1, yy1, yx2, yy2, yconf, ylabel = yolo_box
    color = color_for_class(ylabel)
    if followed:
        thickness, prefix = 3, "FOLLOW"
    else:
        thickness, prefix = 2, "YOLO"
    ydx1 = int(yx1 * sx); ydy1 = int(yy1 * sy)
    ydx2 = int(yx2 * sx); ydy2 = int(yy2 * sy)
    cv2.rectangle(disp, (ydx1, ydy1), (ydx2, ydy2), color, thickness)
    ytxt = f"{prefix} {ylabel} {yconf:.2f}"
    cv2.putText(disp, ytxt, (ydx1, max(ydy1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(disp, ytxt, (ydx1, max(ydy1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_aim_crosshair(disp, aim, sx: float, sy: float) -> None:
    """Draw a red crosshair at the follower's predicted aim point.

    Shows what the camera is steering toward — visible even while the follower
    is coasting (no detection this cycle, so no box is drawn).

    Args:
        disp: BGR display frame (modified in-place).
        aim:  ``(px, py)`` predicted aim point in camera-native pixels.
        sx:   Horizontal scale factor from camera to display pixels.
        sy:   Vertical scale factor from camera to display pixels.
    """
    import cv2

    ax = int(aim[0] * sx); ay = int(aim[1] * sy)
    r = 14
    cv2.line(disp, (ax - r, ay), (ax + r, ay), _FOLLOW_COLOR, 2)
    cv2.line(disp, (ax, ay - r), (ax, ay + r), _FOLLOW_COLOR, 2)
    cv2.circle(disp, (ax, ay), 4, _FOLLOW_COLOR, -1)
