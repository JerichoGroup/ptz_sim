"""HUD overlay for the live PTZ viewer window.

``HudState`` captures a snapshot of all display-relevant pipeline state so
``draw_hud`` does not need to reach back into module-level globals.
"""

__all__ = ["HudState", "draw_hud", "STATE_COL"]

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


STATE_COL = {
    "IDLE":  (200, 200,   0),
    "TRACK": (  0, 255, 100),
}

_DEFAULT_COL = (200, 200, 200)


@dataclass
class HudState:
    """Snapshot of all information the HUD needs; built by the display thread."""
    state:       str
    locked_id:   Optional[int]
    dfps:        float
    det_fps:     float
    zoom_pos:    float
    focal_mm:    float
    fov_deg:     float
    pid_err:     Tuple[float, float]
    pid_kp:      float
    yolo_active: bool
    cam_w:       int
    cam_h:       int
    recording:   bool
    ptz_ready:   bool
    ptz_error:   Optional[str] = field(default=None)


def draw_hud(disp, hud: HudState) -> None:
    """Render the HUD overlay onto ``disp`` in-place.

    Args:
        disp: BGR frame (will be modified in-place).
        hud:  Snapshot of current pipeline state.
    """
    import cv2   # local import so the module is importable without cv2 on the test runner
    import numpy as np

    h, w = disp.shape[:2]
    col  = STATE_COL.get(hud.state, _DEFAULT_COL)
    ex, ey = hud.pid_err

    lock_str = f"  ID={hud.locked_id}" if hud.locked_id is not None else ""
    lines = [
        f"STATE  : {hud.state}{lock_str}",
        f"FPS    : disp {hud.dfps:.0f}  det {hud.det_fps:.0f}",
        f"Zoom   : {hud.zoom_pos:.2f}  ({hud.focal_mm:.0f}mm)  FOV {hud.fov_deg:.1f}°",
        f"PID err: ex={ex:+.3f}  ey={ey:+.3f}",
        f"PID Kp : {hud.pid_kp:.2f}",
        f"YOLO   : {'ACTIVE' if hud.yolo_active else 'idle'}",
        f"Source : {hud.cam_w}x{hud.cam_h}",
    ]
    for i, txt in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1)

    if not hud.ptz_ready:
        err_short = (hud.ptz_error or "unknown error")[:72]
        offline_txt = f"PTZ OFFLINE — {err_short}"
        y_off = 28 + len(lines) * 26
        cv2.putText(disp, offline_txt, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(disp, offline_txt, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)

    # State badge (top-right corner)
    badge = f"  {hud.state}  "
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    bx = w - bw - 12; by = 42
    cv2.rectangle(disp, (bx - 4, by - bh - 6), (bx + bw + 4, by + 6), col, -1)
    cv2.putText(disp, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    if hud.recording:
        cv2.circle(disp, (w - 20, 20), 10, (0, 0, 255), -1)
        cv2.putText(disp, "REC", (w - 55, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Centre crosshair
    cx, cy = w // 2, h // 2
    cv2.line(disp, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
    cv2.line(disp, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

    if hud.state == "TRACK":
        err_x = int(ex * w)
        err_y = int(ey * h)
        cv2.arrowedLine(disp, (cx, cy), (cx + err_x, cy + err_y),
                        (0, 255, 255), 2, tipLength=0.3)

    hint = ("Q=quit  S=shot  V=rec  T/ENTER=lock+zoom  R=unlock+home"
            "  H=save-home  Z/X=zoom  P/O=PID gain  Arrows=pan/tilt")
    cv2.putText(disp, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
    cv2.putText(disp, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
