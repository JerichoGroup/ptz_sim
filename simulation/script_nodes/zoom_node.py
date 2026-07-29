"""Script node: subscribes to /isaac_core/zoom (Float32, normalised 0-1) and
updates the camera focal length so Isaac Sim renders the correct FOV.

Camera: Netz-250 (20x optical, Sony IMX415 1/2.8" sensor)
  sensor: 5.57 x 3.13 mm

The focal length is NOT a linear interpolation between datasheet wide/tele
values, and it is NOT derived from the camera's on-screen magnification readout.
Both overstate the real magnification. Instead the field of view comes from the
pan-shift curve DroneTracker measured on the actual camera
(scripts/calib_zoom_via_pan.py):

    HFoV(Z) = (pan_range_deg / 2) / shift_per_pan(Z)

Because the tracking algorithm centres targets with dx = ex / shift_per_pan(Z),
deriving the rendered FOV from that same curve guarantees a commanded pan delta
moves the image by exactly the fraction the algorithm expects — at every zoom.

KEEP IN SYNC with ptz/netz250.py (this file is loaded standalone by Isaac Sim as
an OmniGraph script node, so it cannot import from the repo).
"""

import math
import os
import threading
import time
import omni.usd
import rclpy
import numpy as np
from std_msgs.msg import Float32
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pxr import UsdGeom


# ── Camera constants (Netz-250) — keep in sync with ptz/netz250.py ───────────

ZOOM_TOPIC        = "/isaac_core/zoom"
CAMERA_PRIM_PATH  = "/Environment/udp_camera/Xform/main_camera_01"

SENSOR_WIDTH_MM  = 5.57      # IMX415 1/2.8"
SENSOR_HEIGHT_MM = 3.13

PAN_RANGE_DEG = 360.0
PAN_DEG_PER_UNIT = PAN_RANGE_DEG / 2.0

# Measured fraction-of-frame shift per ONVIF pan unit, vs normalised zoom.
PAN_SHIFT_CURVE = [
    (0.0, 2.77), (0.1, 3.24), (0.2, 3.92), (0.3, 4.50), (0.4, 6.38), (0.5, 8.31),
]
# On-screen (overstated) magnification — used ONLY to extrapolate the trend
# above the measured range.
OVERLAY_MAG_CURVE = [
    (0.0, 1.0), (0.1, 2.0), (0.2, 4.0), (0.3, 6.0), (0.4, 8.0), (0.5, 10.0),
    (0.6, 12.0), (0.7, 14.0), (0.8, 18.0), (0.9, 23.0), (1.0, 30.0),
]
MEASURED_ZOOM_MAX = 0.5

HORIZONTAL_APERTURE = SENSOR_WIDTH_MM
VERTICAL_APERTURE   = SENSOR_HEIGHT_MM


def _interp(curve, z):
    return float(np.interp(z, [p[0] for p in curve], [p[1] for p in curve]))


def _true_magnification(z):
    z = max(0.0, min(1.0, float(z)))
    base = _interp(PAN_SHIFT_CURVE, 0.0)
    if z <= MEASURED_ZOOM_MAX:
        return _interp(PAN_SHIFT_CURVE, z) / base
    mag_edge = _interp(PAN_SHIFT_CURVE, MEASURED_ZOOM_MAX) / base
    ovl_edge = _interp(OVERLAY_MAG_CURVE, MEASURED_ZOOM_MAX)
    return mag_edge * (_interp(OVERLAY_MAG_CURVE, z) / max(ovl_edge, 1e-6))


def _hfov_deg(z):
    return (PAN_DEG_PER_UNIT / _interp(PAN_SHIFT_CURVE, 0.0)) / _true_magnification(z)


def _focal_length_mm(z):
    return SENSOR_WIDTH_MM / (2.0 * math.tan(math.radians(_hfov_deg(z)) / 2.0))


FOCAL_LENGTH_WIDE = _focal_length_mm(0.0)

# ── Diagnostics ──────────────────────────────────────────────────────────────
# This node runs INSIDE Isaac Sim, whose stdout is hidden unless
# show_isaac_logs=True. Mirror to a file so zoom problems can be diagnosed with
#   tail -f /tmp/ptz_sim_zoom.log
# without turning on the whole Isaac firehose.
_DIAG_PATH = os.environ.get("PTZ_ZOOM_LOG", "/tmp/ptz_sim_zoom.log")


def _diag(msg):
    line = f"[zoom_node] {msg}"
    print(line, flush=True)
    try:
        with open(_DIAG_PATH, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    except Exception:
        pass


# ── ROS2 subscriber ───────────────────────────────────────────────────────────

class ZoomSubscriber:
    def __init__(self):
        self.node = rclpy.create_node("zoom_script_node")
        try:
            self.node.declare_parameter("use_sim_time", True)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.zoom = None
        self._subscriber = None
        self._spinning = False

    def subscribe(self):
        if self._subscriber is None:
            self._subscriber = self.node.create_subscription(
                Float32, ZOOM_TOPIC, self._callback, self.qos_profile
            )
        if not self._spinning:
            threading.Thread(target=self._spin, daemon=True).start()
            self._spinning = True

    def _callback(self, msg):
        self.zoom = float(msg.data)

    def _spin(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self.node)
        executor.spin()


# ── Script node interface ─────────────────────────────────────────────────────

def setup(db):
    _diag(f"setup() starting — diagnostics also at {_DIAG_PATH}")
    if not rclpy.ok():
        try:
            rclpy.init()
        except Exception as e:
            _diag(f"FATAL: rclpy.init() failed: {e}")
            raise RuntimeError("zoom_node: failed to init rclpy") from e

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

    if not camera_prim.IsValid():
        _diag(f"FATAL: camera prim not found at '{CAMERA_PRIM_PATH}' — zoom will "
              f"NOT work. This path must match the udp_camera USD "
              f"(sim_utils.OPTIONAL_USDS['com_udp']).")
        db.internal_state.focal_attr = None
    else:
        cam = UsdGeom.Camera(camera_prim)

        # Fix sensor geometry once at startup so FOV is correct at every zoom level
        cam.GetHorizontalApertureAttr().Set(float(HORIZONTAL_APERTURE))
        cam.GetVerticalApertureAttr().Set(float(VERTICAL_APERTURE))
        cam.GetFocalLengthAttr().Set(float(FOCAL_LENGTH_WIDE))   # start at wide

        db.internal_state.focal_attr = cam.GetFocalLengthAttr()
        _diag(f"camera ready (Netz-250) — HA={HORIZONTAL_APERTURE:.3f} mm  "
              f"VA={VERTICAL_APERTURE:.3f} mm  FL={FOCAL_LENGTH_WIDE:.2f} mm "
              f"(HFoV {_hfov_deg(0.0):.1f}°) → {_focal_length_mm(1.0):.2f} mm "
              f"(HFoV {_hfov_deg(1.0):.1f}°)")

    sub = ZoomSubscriber()
    sub.subscribe()
    db.internal_state.zoom_subscriber = sub
    db.internal_state.last_zoom = None
    db.internal_state.n_msgs = 0
    db.internal_state.last_report = 0.0
    _diag(f"setup complete — subscribed to {ZOOM_TOPIC}")


def compute(db):
    st = db.internal_state
    zoom = st.zoom_subscriber.zoom

    # Heartbeat every ~5 s: makes "no zoom messages arriving" obvious instead of
    # looking identical to "zoom arriving but not applied".
    now = time.time()
    if now - getattr(st, "last_report", 0.0) > 5.0:
        st.last_report = now
        if st.focal_attr is None:
            _diag("WARNING: no camera prim — zoom cannot be applied")
        elif zoom is None:
            _diag(f"WARNING: no messages on {ZOOM_TOPIC} yet "
                  f"(is ptz_sim.py running and publishing?)")
        else:
            _diag(f"alive — last zoom={zoom:.3f}, applied={st.last_zoom}, "
                  f"msgs={st.n_msgs}")

    if zoom is None or zoom == st.last_zoom:
        return True
    if st.focal_attr is None:
        return True

    zoom = max(0.0, min(1.0, zoom))
    focal_length = _focal_length_mm(zoom)
    try:
        st.focal_attr.Set(float(focal_length))
    except Exception as e:
        _diag(f"ERROR setting focalLength: {e}")
        return True
    st.last_zoom = zoom
    st.n_msgs = getattr(st, "n_msgs", 0) + 1

    _diag(f"zoom={zoom:.3f}  mag={_true_magnification(zoom):.2f}x  "
          f"FL={focal_length:.2f} mm  HFoV={_hfov_deg(zoom):.2f}°")

    return True


def cleanup(db):
    try:
        db.internal_state.zoom_subscriber.node.destroy_node()
    except Exception:
        pass
    db.internal_state.focal_attr = None
    db.internal_state.zoom_subscriber = None
    db.internal_state.last_zoom = None
