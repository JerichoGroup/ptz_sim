"""dronetracker/ptz/follow.py

Pure, time-based constant-velocity prediction filter for a single tracked
target's pixel center.  Feeds the live PTZ follow loop but has no dependencies
on PTZ hardware, YOLO, cv2, or torch — fully unit-testable in isolation.

Why constant-velocity rather than Norfair's Kalman
---------------------------------------------------
Norfair's Kalman filter is frame-indexed (px/frame).  In TRACK state YOLO runs
on a time-throttled cadence (monitor_interval_s ~0.1 s) with variable latency
and frame gaps during zoom.  Feeding a frame-indexed tracker irregular updates
produces wrong velocity units.  A time-based EMA with explicit wall-clock
timestamps sidesteps this entirely.
"""
from __future__ import annotations

__all__ = ["TargetFollower"]


class TargetFollower:
    """Constant-velocity EMA prediction filter for one target's pixel center.

    Parameters (from cfg)
    ---------------------
    follow_vel_ema      : float — EMA alpha for velocity smoothing (e.g. 0.5).
                          Higher = more inertia / less reactive to new samples.
    follow_coast_decay  : float — per-coast() velocity multiplier (e.g. 0.8).
                          Applied each time a YOLO miss is signalled.
    follow_coast_frames : int   — not enforced here (reserved for caller logic).

    State attributes
    ----------------
    pos           : (float, float)  last known pixel center (cx, cy)
    vel           : (float, float)  estimated velocity in px/s
    last_meas_t   : float           wall-clock timestamp of last measurement
    coast_n       : int             consecutive coast() calls since last measure()
    initialized   : bool            False until the first measure() call
    """

    def __init__(self, cfg):
        self._alpha = float(cfg.follow_vel_ema)
        self._decay = float(cfg.follow_coast_decay)

        # Public state (read by callers / tests)
        self.pos: tuple[float, float] = (0.0, 0.0)
        self.vel: tuple[float, float] = (0.0, 0.0)
        self.last_meas_t: float = 0.0
        self.coast_n: int = 0
        self.initialized: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def measure(self, cx: float, cy: float, t: float) -> None:
        """Record a new YOLO detection at pixel center (cx, cy) at time t."""
        if not self.initialized:
            self.pos = (float(cx), float(cy))
            self.vel = (0.0, 0.0)
            self.last_meas_t = float(t)
            self.initialized = True
            self.coast_n = 0
            return

        dt = max(1e-4, t - self.last_meas_t)

        vx_inst = (cx - self.pos[0]) / dt
        vy_inst = (cy - self.pos[1]) / dt

        alpha = self._alpha
        self.vel = (
            alpha * self.vel[0] + (1.0 - alpha) * vx_inst,
            alpha * self.vel[1] + (1.0 - alpha) * vy_inst,
        )

        self.pos = (float(cx), float(cy))
        self.last_meas_t = float(t)
        self.coast_n = 0

    def predict(self, t: float, lead_s: float, fw: int, fh: int) -> tuple[float, float]:
        """Dead-reckon target position at time (t + lead_s).

        Parameters
        ----------
        t      : current wall-clock time
        lead_s : look-ahead in seconds (accounts for PTZ latency)
        fw, fh : frame width / height in pixels (used for clamping)

        Returns
        -------
        (px, py) clamped to [0, fw] × [0, fh]
        """
        if not self.initialized:
            return (fw / 2.0, fh / 2.0)

        dt_since = max(0.0, t - self.last_meas_t)
        horizon = dt_since + lead_s

        px = self.pos[0] + self.vel[0] * horizon
        py = self.pos[1] + self.vel[1] * horizon

        px = max(0.0, min(float(fw), px))
        py = max(0.0, min(float(fh), py))

        return (px, py)

    def coast(self) -> None:
        """Signal a YOLO miss — decay velocity to prevent runaway extrapolation."""
        self.vel = (
            self.vel[0] * self._decay,
            self.vel[1] * self._decay,
        )
        self.coast_n += 1

    def reset(self) -> None:
        """Clear all state back to uninitialized."""
        self.pos = (0.0, 0.0)
        self.vel = (0.0, 0.0)
        self.last_meas_t = 0.0
        self.coast_n = 0
        self.initialized = False
