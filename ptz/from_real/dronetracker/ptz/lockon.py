# -*- coding: utf-8 -*-
"""Lock-on geometry for the IDLE → TRACK transition.

Pure math: given the best track's estimate/velocity, the frame size, the current
detect-loop FPS and zoom position, compute the relative pan/tilt move (FOV-space
``dx``/``dy``), the zoom target, and the velocity-lead bookkeeping used in the
``[TRACK] locked`` log line.  No PTZ I/O, no state mutation — the caller
(``DetectWorker._step_idle``) owns ``center_and_zoom`` and the state transition.
"""

__all__ = ["LockonCommand", "compute_lockon_command"]

from dataclasses import dataclass

import numpy as np

from dronetracker.ptz.transform import _F_WIDE, _F_TELE


@dataclass
class LockonCommand:
    """Result of the lock-on geometry computation.

    ``dx``, ``dy``, ``zoom_pos_target`` drive ``center_and_zoom``; the remaining
    fields are diagnostics for the ``[TRACK] locked`` log line.
    """
    dx:              float
    dy:              float
    zoom_pos_target: float
    zoom_pos_now:    float
    aim_x:           float   # leaded target pixel x
    aim_y:           float   # leaded target pixel y
    ex:              float   # normalised x offset from frame center
    ey:              float   # normalised y offset from frame center
    lead_frames:     float
    vx:              float
    vy:              float


def compute_lockon_command(best_obj, fw, fh, fps_val, zoom_pos_now, frozen, zoom) -> LockonCommand:
    """Compute the relative PTZ move + zoom target to lock onto ``best_obj``.

    ``frozen`` / ``zoom`` are the ``FrozenConfig`` / ``ZoomConfig`` groups.
    ``fps_val`` is the detect-loop FPS (falls back to 25 fps below 1.0).
    """
    # ── 1. Zoom target ────────────────────────────────────────────────────────
    f_now           = zoom_pos_now * (_F_TELE - _F_WIDE) + _F_WIDE
    zoom_pos_target = float(np.clip(
        (frozen.zoom_factor * f_now - _F_WIDE) / (_F_TELE - _F_WIDE),
        0.0, zoom.max_pos))
    zoom_burst_s = max(0.4, abs(zoom_pos_target - zoom_pos_now) * zoom.full_travel_s)
    settle_s     = max(frozen.move_settle_s, zoom_burst_s)

    # ── 2. Lead the target ──────────────────────────────────────────────────────
    fps = fps_val if fps_val > 1.0 else 25.0
    px, py = float(best_obj.estimate[0][0]), float(best_obj.estimate[0][1])
    try:
        vx = float(best_obj.estimate_velocity[0][0])
        vy = float(best_obj.estimate_velocity[0][1])
    except Exception:
        vx, vy = 0.0, 0.0
    lead_frames = settle_s * fps
    # Cap the lead displacement: a long zoom-burst settle × a noisy velocity can
    # otherwise fling the aim far off the target (clipped to a frame edge), so the
    # camera zooms onto empty space.  Keep the aim within max_lead_frac of the
    # frame from the target's current position so it stays inside the post-zoom FOV.
    max_lead_x = frozen.lockon_max_lead_frac * fw
    max_lead_y = frozen.lockon_max_lead_frac * fh
    lead_x = float(np.clip(vx * lead_frames, -max_lead_x, max_lead_x))
    lead_y = float(np.clip(vy * lead_frames, -max_lead_y, max_lead_y))
    px = float(np.clip(px + lead_x, 0.0, fw))
    py = float(np.clip(py + lead_y, 0.0, fh))

    # ── 3. Pixel offset → FOV-space PTZ move ────────────────────────────────────
    ex = px / fw - 0.5
    ey = py / fh - 0.5
    dx = float(np.clip(frozen.fov_sign_x * frozen.fov_gain * 2.0 * ex, -1.0, 1.0))
    dy = float(np.clip(frozen.fov_sign_y * frozen.fov_gain * 2.0 * ey, -1.0, 1.0))

    return LockonCommand(
        dx=dx, dy=dy, zoom_pos_target=zoom_pos_target, zoom_pos_now=zoom_pos_now,
        aim_x=px, aim_y=py, ex=ex, ey=ey, lead_frames=lead_frames, vx=vx, vy=vy,
    )
