"""PTZ coordinate transformation for Norfair ego-motion compensation.

Anchors tracked coordinates to the camera pose at the moment of track-lock so
Norfair's Kalman filter sees only the *drone's* true motion, not
drone-motion + camera-ego-motion.
"""

# Note: private names (_zoom_mag, _F_WIDE, _F_TELE, _SENSOR_W) are intentionally
# omitted from __all__ — they are internal details used by ptz_controller and the
# pipeline, and are explicitly imported by name wherever needed.
__all__ = ["PTZTransformation"]

import numpy as np
from norfair.camera_motion import CoordinatesTransformation

# UNV IPC6852ER-X45-VF 45× lens constants — update these when swapping the lens.
_F_WIDE   = 5.7     # focal length at wide end (mm)
_F_TELE   = 256.5   # focal length at tele end (mm)
_SENSOR_W = 7.18    # sensor width (mm), used for horizontal FOV


def _zoom_mag(zoom_pos: float) -> float:
    """Camera optical magnification at zoom_pos ∈ [0, 1].

    Returns 1.0 at the wide end and ~45 at the tele end.
    """
    f = zoom_pos * (_F_TELE - _F_WIDE) + _F_WIDE
    return f / _F_WIDE


class PTZTransformation(CoordinatesTransformation):
    """Norfair coord transform anchored to the camera pose at track-lock time.

    Accounts for pan/tilt (image translation) and zoom (scale about frame centre).
    Norfair calls ``rel_to_abs`` when ingesting detections and ``abs_to_rel`` when
    returning ``obj.estimate``, so ``obj.estimate`` continues to return relative
    (image-pixel) coordinates — no PID / drawing changes are needed at call sites.

    Derivation for a static world point:
        p_cur = center + (f_now / f_ref) * (p_ref − center) + shift
      → rel_to_abs(p_cur) = center + (f_ref / f_now) * (p_cur − shift − center)
      → abs_to_rel(p_ref) = center + (p_ref − center) / (f_ref / f_now) + shift

    where:
        shift   = [k_pan·mag·Δpan,  k_tilt·mag·Δtilt]
        k_pan / k_tilt  are calibrated online (pixels per pose-unit per mag)
        scale   = f_ref / f_now  (1.0 at lock, < 1.0 when zoomed in further)
    """

    def __init__(self, center: np.ndarray, scale: float, shift: np.ndarray):
        self.c = center        # [frame_width/2, frame_height/2]
        self.s = float(scale)  # f_ref / f_now at the time of the update
        self.d = shift         # [dx_px, dy_px] camera-induced pixel translation

    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        return self.c + self.s * (points - self.d - self.c)

    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        return self.c + (points - self.c) / self.s + self.d
