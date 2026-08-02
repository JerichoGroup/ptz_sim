"""Netz-250 camera model — the physical behaviour the simulator reproduces.

This is the ONE place the host describes the camera it is pretending to be.
Every number here mirrors the ``Netz-250`` entry in ``CAMERA_PRESETS``
(DroneTracker: ``dronetracker/config/schema.py``).  If the algorithm side
changes a lens constant, change it here too.

Why the geometry is derived from a measured curve rather than the datasheet
-------------------------------------------------------------------------
The camera's on-screen overlay reports magnification up to 30x, but that number
is not the true optical magnification — DroneTracker's ``calib_zoom_via_pan.py``
measured the real thing by panning a known ONVIF delta and observing how far the
scene actually moved:

    mag(Z) = shift_per_pan(Z) / shift_per_pan(wide)

That measurement says the true optical magnification only reaches ~3x at
zoom 0.5, while the overlay claims 10x.  The algorithm centres the target using
``dx = ex / shift_per_pan(Z)``, so if the simulator rendered the overlay's
magnification instead of the measured one, every lock-on would badly overshoot.

So the simulator derives its field of view directly from the same measured
curve the algorithm uses:

    HFoV(Z) = (pan_range_deg / 2) / shift_per_pan(Z)

Pan/tilt then stay LINEAR in ONVIF units (180 deg per unit of pan, 45 deg per
unit of tilt), and a commanded pan delta produces exactly the image shift the
algorithm expects, at every zoom level.  No "proportional pan" fudge needed.
"""

import math

import numpy as np

# ── Mechanical ranges: ONVIF pan/tilt [-1,+1] spans this many degrees ────────
PAN_RANGE_DEG  = 360.0
TILT_RANGE_DEG = 90.0
# Degrees of gimbal movement per unit of ONVIF pan/tilt (linear).
PAN_DEG_PER_UNIT  = PAN_RANGE_DEG  / 2.0    # 180 deg/unit
TILT_DEG_PER_UNIT = TILT_RANGE_DEG / 2.0    # 45 deg/unit

# ── Slew speeds (datasheet) at full ContinuousMove velocity ──────────────────
PAN_SPEED_DEG_S  = 24.5
TILT_SPEED_DEG_S = 15.0

# ── Zoom ─────────────────────────────────────────────────────────────────────
# ONVIF reports/accepts zoom in this RAW range; the algorithm normalises it
# to [0,1] internally (raw -1 -> 0.0, raw +1 -> 1.0).
ZOOM_RAW_MIN = -1.0
ZOOM_RAW_MAX =  1.0
ZOOM_FULL_TRAVEL_S = 4.0        # seconds for the full zoom range, end to end

# ── Sensor / lens ────────────────────────────────────────────────────────────
SENSOR_WIDTH_MM  = 5.57         # IMX415 1/2.8"
SENSOR_HEIGHT_MM = 3.13

# Measured shift-per-ONVIF-unit vs zoom (DroneTracker calib_zoom_via_pan.py).
# Fraction of the frame the scene moves per unit of pan/tilt at that zoom.
PAN_SHIFT_CURVE = [
    (0.0, 2.77), (0.1, 3.24), (0.2, 3.92), (0.3, 4.50), (0.4, 6.38), (0.5, 8.31),
]
TILT_SHIFT_CURVE = [
    (0.0, 1.44), (0.1, 1.57), (0.2, 1.82), (0.3, 2.32), (0.4, 3.06), (0.5, 3.86),
]

# The camera's on-screen overlay magnification.  NOT the true optical
# magnification (see the module docstring) — used only to extrapolate the shape
# of the zoom response ABOVE the measured range.
OVERLAY_MAG_CURVE = [
    (0.0, 1.0), (0.1, 2.0), (0.2, 4.0), (0.3, 6.0), (0.4, 8.0), (0.5, 10.0),
    (0.6, 12.0), (0.7, 14.0), (0.8, 18.0), (0.9, 23.0), (1.0, 30.0),
]

# Highest zoom the measured curves cover.  Above this the simulator has to
# extrapolate — see true_magnification().
MEASURED_ZOOM_MAX = 0.5

# ── AbsoluteMove quantisation quirk ──────────────────────────────────────────
# Measured on the real camera: AbsoluteMove snaps pan/tilt onto a coarse grid of
# ~0.02 ONVIF units (~3.6 deg of pan), so small deltas either round away to
# nothing or overshoot.  This is the entire reason DroneTracker added
# PTZController.center_fine() (RelativeMove, which is accurate ~1:1).
# Reproduced here on purpose: without it the simulator would centre perfectly at
# high zoom and hide a bug class that is very real on the hardware.
ABSOLUTE_MOVE_GRID = 0.02

# Real RelativeMove is near-exact (0.005 commanded -> 0.0049 measured).
RELATIVE_MOVE_ACCURACY = 1.0


def _interp(curve, z):
    """np.interp over a [(x, y), ...] curve; clamps outside the range."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    return float(np.interp(z, xs, ys))


def shift_per_pan(zoom_norm: float) -> float:
    """Measured fraction-of-frame the scene shifts per ONVIF pan unit."""
    return _interp(PAN_SHIFT_CURVE, zoom_norm)


def shift_per_tilt(zoom_norm: float) -> float:
    """Measured fraction-of-frame the scene shifts per ONVIF tilt unit.

    REFERENCE DATA, not currently in the render path: Isaac derives the vertical
    FoV from the horizontal one plus the resolution aspect ratio, so only
    shift_per_pan feeds hfov_deg().  Kept because TILT_SHIFT_CURVE is measured off
    the real camera and cannot be regenerated without hardware access.
    """
    return _interp(TILT_SHIFT_CURVE, zoom_norm)


def true_magnification(zoom_norm: float) -> float:
    """True optical magnification at normalised zoom, relative to the wide end.

    Within the measured range this is ``shift_per_pan(Z)/shift_per_pan(0)``.

    Above ``MEASURED_ZOOM_MAX`` there is no measurement, so the overlay curve's
    SHAPE is used to continue the trend, anchored to the last measured point:

        mag(Z) = mag(0.5) * overlay(Z) / overlay(0.5)

    That keeps zooming beyond 0.5 doing something sensible instead of flat-lining
    (np.interp would clamp), while never trusting the overlay's absolute values.
    Extend PAN_SHIFT_CURVE / TILT_SHIFT_CURVE with real measurements above 0.5 to
    remove the guesswork.
    """
    z = max(0.0, min(1.0, float(zoom_norm)))
    base = shift_per_pan(0.0)
    if z <= MEASURED_ZOOM_MAX:
        return shift_per_pan(z) / base
    mag_at_edge = shift_per_pan(MEASURED_ZOOM_MAX) / base
    ovl_edge = _interp(OVERLAY_MAG_CURVE, MEASURED_ZOOM_MAX)
    ovl_now = _interp(OVERLAY_MAG_CURVE, z)
    return mag_at_edge * (ovl_now / max(ovl_edge, 1e-6))


def hfov_deg(zoom_norm: float) -> float:
    """Horizontal field of view (degrees) at normalised zoom.

    Anchored to the measured pan curve so a commanded pan delta shifts the image
    by exactly the fraction the algorithm assumes.
    """
    return (PAN_DEG_PER_UNIT / shift_per_pan(0.0)) / true_magnification(zoom_norm)


def focal_length_mm(zoom_norm: float) -> float:
    """Focal length that renders ``hfov_deg(zoom_norm)`` for this sensor."""
    hfov = math.radians(hfov_deg(zoom_norm))
    return SENSOR_WIDTH_MM / (2.0 * math.tan(hfov / 2.0))


def normalize_zoom(raw: float) -> float:
    """Camera-native RAW ONVIF zoom -> internal [0,1] (matches PTZController)."""
    span = ZOOM_RAW_MAX - ZOOM_RAW_MIN
    if span == 0:
        return 0.0
    return (float(raw) - ZOOM_RAW_MIN) / span


def denormalize_zoom(pos: float) -> float:
    """Internal [0,1] -> camera-native RAW ONVIF zoom (matches PTZController)."""
    span = ZOOM_RAW_MAX - ZOOM_RAW_MIN
    return ZOOM_RAW_MIN + float(pos) * span


def quantize_absolute(value: float) -> float:
    """Snap an AbsoluteMove pan/tilt target onto the camera's coarse grid."""
    if ABSOLUTE_MOVE_GRID <= 0:
        return value
    return round(value / ABSOLUTE_MOVE_GRID) * ABSOLUTE_MOVE_GRID


def clamp_pan(units: float) -> float:
    """Pan is continuous but ONVIF-normalised; keep it inside [-1,+1]."""
    return max(-1.0, min(1.0, float(units)))


def clamp_tilt(units: float) -> float:
    """Tilt is mechanically limited; ONVIF-normalised to [-1,+1]."""
    return max(-1.0, min(1.0, float(units)))
