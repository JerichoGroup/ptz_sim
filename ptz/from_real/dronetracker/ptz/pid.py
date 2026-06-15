"""Dual-axis PID controller for PTZ pan/tilt velocity commands."""

__all__ = ["PIDController"]

import time
import numpy as np


class PIDController:
    """Dual-axis PID for PTZ pan/tilt.

    Input:  normalised drone centre in [0, 1] × [0, 1] (fraction of frame).
    Output: ONVIF velocity in [-1, +1] on each axis.

    Positive vx → pan right.
    Positive vy → tilt up.
    """

    def __init__(self, kp, ki, kd, max_out, deadband, i_limit):
        self.kp       = kp
        self.ki       = ki
        self.kd       = kd
        self.max_out  = max_out
        self.deadband = deadband
        self.i_limit  = i_limit

        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.int_x      = 0.0
        self.int_y      = 0.0
        self.last_t     = None

    def reset(self):
        """Clear integrators and derivative history."""
        self.prev_err_x = self.prev_err_y = 0.0
        self.int_x      = self.int_y = 0.0
        self.last_t     = None

    def compute(self, drone_cx_n: float, drone_cy_n: float):
        """Return (vx, vy) ONVIF velocities.

        Args:
            drone_cx_n: Drone centre x as fraction of frame width  (0 = left,  1 = right).
            drone_cy_n: Drone centre y as fraction of frame height (0 = top,   1 = bottom).

        Returns:
            (vx, vy): Clamped to ±max_out.  Both zero when inside the deadband.
        """
        now      = time.time()
        dt       = 0.033 if self.last_t is None else max(0.001, now - self.last_t)
        self.last_t = now

        ex =  (drone_cx_n - 0.5)   # positive → drone right  → pan right
        ey = -(drone_cy_n - 0.5)   # positive → drone above  → tilt up

        if abs(ex) < self.deadband and abs(ey) < self.deadband:
            # Decay integrators in the deadband to prevent windup accumulation.
            self.int_x *= 0.9
            self.int_y *= 0.9
            return 0.0, 0.0

        self.int_x = np.clip(self.int_x + ex * dt, -self.i_limit, self.i_limit)
        self.int_y = np.clip(self.int_y + ey * dt, -self.i_limit, self.i_limit)

        dx = (ex - self.prev_err_x) / dt
        dy = (ey - self.prev_err_y) / dt
        self.prev_err_x = ex
        self.prev_err_y = ey

        vx = self.kp * ex + self.ki * self.int_x + self.kd * dx
        vy = self.kp * ey + self.ki * self.int_y + self.kd * dy
        return (
            float(np.clip(vx, -self.max_out, self.max_out)),
            float(np.clip(vy, -self.max_out, self.max_out)),
        )
