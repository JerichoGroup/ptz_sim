"""Script node: subscribes to /isaac_core/zoom (Float32, normalised 0-1) and
updates the camera focal length so Isaac Sim renders the correct FOV.

Camera: Netz-250 (Sony IMX415 1/2.8" sensor, 20x optical)
  sensor: 5.57 x 3.13 mm

Field of view is NOT a linear interpolation between datasheet wide/tele focal
lengths, and it is NOT taken from the camera's on-screen magnification readout —
both overstate the real optics.  It comes from the pan-shift curve DroneTracker
measured on the actual hardware (scripts/calib_zoom_via_pan.py):

    HFoV(Z) = (pan_range_deg / 2) / shift_per_pan(Z)

The tracker centres targets with dx = ex / shift_per_pan(Z), so deriving the
rendered FOV from that same curve guarantees a commanded pan delta moves the
image by exactly the fraction the algorithm expects, at every zoom level.

KEEP THE CONSTANTS IN SYNC with ptz/netz250.py.  This file is exec'd standalone
by Isaac's script node, so it cannot import from the repo.

─────────────────────────────────────────────────────────────────────────────
IMPORTANT — how omni.graph.scriptnode runs this file (read before editing):

  exec(code_object)                              # <- this module body runs
  ...
  compute_fn.__globals__.update(script_context)  # <- names become visible HERE

Names defined here are injected into the functions' globals only AFTER the
module body has finished executing.  Therefore:

  * DO NOT call any function defined in this file at module level.  It will raise
    NameError on the first global it touches, the exec aborts, no setup/compute
    are extracted, and the node silently does nothing at all — including no log
    output, because the logging call is itself what died.
  * Module level may only do: imports, literals, and calls on IMPORTED modules
    (e.g. math.tan(...)), which resolve in the exec frame directly.
  * Helper functions are fine — just call them only from setup()/compute()/
    cleanup(), which run after the globals are patched.

This is exactly how the zoom silently broke once before.  Keep it in mind.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import threading
import time
import omni.usd
import rclpy
from std_msgs.msg import Float32
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pxr import UsdGeom


# ── Camera constants (Netz-250) — keep in sync with ptz/netz250.py ───────────

ZOOM_TOPIC        = "/isaac_core/zoom"
CAMERA_PRIM_PATH  = "/Environment/udp_camera/Xform/main_camera_01"

# Diagnostics file.  Isaac's stdout is hidden unless show_isaac_logs=True, so
# mirror the important lines here:  tail -f /tmp/ptz_sim_zoom.log
ZOOM_LOG_PATH = "/tmp/ptz_sim_zoom.log"

SENSOR_WIDTH_MM  = 5.57      # IMX415 1/2.8"
SENSOR_HEIGHT_MM = 3.13

# ONVIF pan [-1,+1] spans 360 deg, so 180 deg per unit.
PAN_DEG_PER_UNIT = 180.0

# Measured fraction-of-frame shift per ONVIF pan unit, vs normalised zoom.
PAN_SHIFT_CURVE = [
    (0.0, 2.77), (0.1, 3.24), (0.2, 3.92), (0.3, 4.50), (0.4, 6.38), (0.5, 8.31),
]
# The camera's on-screen (overstated) magnification.  Used ONLY to continue the
# trend above the measured range — never for absolute values.
OVERLAY_MAG_CURVE = [
    (0.0, 1.0), (0.1, 2.0), (0.2, 4.0), (0.3, 6.0), (0.4, 8.0), (0.5, 10.0),
    (0.6, 12.0), (0.7, 14.0), (0.8, 18.0), (0.9, 23.0), (1.0, 30.0),
]
MEASURED_ZOOM_MAX = 0.5      # highest zoom the curves actually cover

# Isaac sensor geometry — the physical sensor, set once in setup().
HORIZONTAL_APERTURE = SENSOR_WIDTH_MM
VERTICAL_APERTURE   = SENSOR_HEIGHT_MM


# ── Helpers (safe to call from setup/compute, NOT from module level) ─────────

def _log(msg):
    """Print and append to ZOOM_LOG_PATH.  Never raises."""
    line = "[zoom_node] " + str(msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(ZOOM_LOG_PATH, "a") as fh:
            fh.write(time.strftime("%H:%M:%S") + " " + line + "\n")
    except Exception:
        pass


def _interp(curve, z):
    """Linear interpolation over [(x, y), ...], clamping outside the range.

    Pure Python: every third-party import is another way for this node to die
    silently inside the script-node sandbox.
    """
    z = float(z)
    if z <= curve[0][0]:
        return float(curve[0][1])
    if z >= curve[-1][0]:
        return float(curve[-1][1])
    for i in range(1, len(curve)):
        x1, y1 = curve[i]
        if z <= x1:
            x0, y0 = curve[i - 1]
            if x1 == x0:
                return float(y1)
            return float(y0 + (z - x0) / (x1 - x0) * (y1 - y0))
    return float(curve[-1][1])


def _true_magnification(z):
    """True optical magnification at normalised zoom, relative to the wide end.

    Inside the measured range this is shift_per_pan(Z)/shift_per_pan(0).  Above
    MEASURED_ZOOM_MAX there is no measurement, so the overlay curve's SHAPE
    continues the trend, anchored to the last real data point.  Extend
    PAN_SHIFT_CURVE with measurements above 0.5 to remove the guesswork.
    """
    z = max(0.0, min(1.0, float(z)))
    base = _interp(PAN_SHIFT_CURVE, 0.0)
    if z <= MEASURED_ZOOM_MAX:
        return _interp(PAN_SHIFT_CURVE, z) / base
    mag_edge = _interp(PAN_SHIFT_CURVE, MEASURED_ZOOM_MAX) / base
    ovl_edge = _interp(OVERLAY_MAG_CURVE, MEASURED_ZOOM_MAX)
    return mag_edge * (_interp(OVERLAY_MAG_CURVE, z) / max(ovl_edge, 1e-6))


def _hfov_deg(z):
    """Horizontal field of view (degrees) at normalised zoom."""
    return (PAN_DEG_PER_UNIT / _interp(PAN_SHIFT_CURVE, 0.0)) / _true_magnification(z)


def _focal_length_mm(z):
    """Focal length that renders _hfov_deg(z) on this sensor."""
    return SENSOR_WIDTH_MM / (2.0 * math.tan(math.radians(_hfov_deg(z)) / 2.0))


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
        self.n_msgs = 0
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
        self.n_msgs += 1

    def _spin(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self.node)
        executor.spin()


# ── Script node interface ─────────────────────────────────────────────────────

def setup(db):
    _log("setup() starting")

    if not rclpy.ok():
        try:
            rclpy.init()
        except Exception as e:
            _log("FATAL: rclpy.init() failed: %s" % e)
            raise RuntimeError("zoom_node: failed to init rclpy") from e

    fl_wide = _focal_length_mm(0.0)

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

    if not camera_prim.IsValid():
        _log("FATAL: camera prim not found at '%s' — zoom cannot be applied. "
             "This must match sim_utils.OPTIONAL_USDS['com_udp']."
             % CAMERA_PRIM_PATH)
        db.internal_state.focal_attr = None
    else:
        cam = UsdGeom.Camera(camera_prim)

        # Fix sensor geometry once at startup so FOV is correct at every zoom level
        cam.GetHorizontalApertureAttr().Set(float(HORIZONTAL_APERTURE))
        cam.GetVerticalApertureAttr().Set(float(VERTICAL_APERTURE))
        cam.GetFocalLengthAttr().Set(float(fl_wide))   # start at wide

        db.internal_state.focal_attr = cam.GetFocalLengthAttr()
        _log("camera ready (Netz-250) — HA=%.3f mm VA=%.3f mm  "
             "FL %.2f mm (HFoV %.1f deg) -> %.2f mm (HFoV %.1f deg)"
             % (HORIZONTAL_APERTURE, VERTICAL_APERTURE,
                fl_wide, _hfov_deg(0.0),
                _focal_length_mm(1.0), _hfov_deg(1.0)))

    sub = ZoomSubscriber()
    sub.subscribe()
    db.internal_state.zoom_subscriber = sub
    db.internal_state.last_zoom = None
    db.internal_state.last_report = 0.0
    _log("setup complete — subscribed to %s" % ZOOM_TOPIC)


def compute(db):
    st = db.internal_state
    zoom = st.zoom_subscriber.zoom

    # Heartbeat every ~5 s so "no messages arriving" is distinguishable from
    # "messages arriving but not applied" and from "node not running at all".
    now = time.time()
    if now - getattr(st, "last_report", 0.0) > 5.0:
        st.last_report = now
        if st.focal_attr is None:
            _log("WARNING: no camera prim — zoom cannot be applied")
        elif zoom is None:
            _log("WARNING: no messages on %s yet (is ptz_sim.py publishing?)"
                 % ZOOM_TOPIC)
        else:
            _log("alive — zoom=%.3f applied=%s msgs=%d"
                 % (zoom, st.last_zoom, st.zoom_subscriber.n_msgs))

    if zoom is None or zoom == st.last_zoom:
        return True
    if st.focal_attr is None:
        return True

    zoom = max(0.0, min(1.0, zoom))
    focal_length = _focal_length_mm(zoom)
    try:
        st.focal_attr.Set(float(focal_length))
    except Exception as e:
        _log("ERROR setting focalLength: %s" % e)
        return True
    st.last_zoom = zoom

    _log("zoom=%.3f  mag=%.2fx  FL=%.2f mm  HFoV=%.2f deg"
         % (zoom, _true_magnification(zoom), focal_length, _hfov_deg(zoom)))

    return True


def cleanup(db):
    try:
        db.internal_state.zoom_subscriber.node.destroy_node()
    except Exception:
        pass
    db.internal_state.focal_attr = None
    db.internal_state.zoom_subscriber = None
    db.internal_state.last_zoom = None
