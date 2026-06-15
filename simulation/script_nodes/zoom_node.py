"""Script node: subscribes to /isaac_core/zoom (Float32, normalised 0-1) and
updates the camera focal length so Isaac Sim renders the correct FOV.

Camera: UNV IPC6852ER-X45-VF (45x optical zoom)
  Wide-end: FL = 5.7 mm,   HFoV = 59.5°
  Tele-end: FL = 256.5 mm, HFoV ~= 1.45°  (datasheet says 2.2° — small
            discrepancy due to lens distortion/rounding in the spec sheet)

Sensor horizontal aperture is fixed at 6.504 mm, derived from the wide-end
numbers: 2 × 5.7 × tan(59.5°/2).  Focal length is linearly interpolated
from 5.7 to 256.5 mm as zoom goes 0 → 1.
"""

import math
import threading
import omni.usd
import rclpy
from std_msgs.msg import Float32
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pxr import UsdGeom


# ── Camera constants (IPC6852ER-X45-VF) ──────────────────────────────────────

ZOOM_TOPIC        = "/isaac_core/zoom"
CAMERA_PRIM_PATH  = "/Environment/udp_camera/Xform/main_camera_01"

FOCAL_LENGTH_WIDE = 5.7      # mm
FOCAL_LENGTH_TELE = 256.5    # mm

# Sensor size derived from wide-end: 2 × FL_wide × tan(HFoV_wide / 2)
HORIZONTAL_APERTURE = 2 * FOCAL_LENGTH_WIDE * math.tan(math.radians(59.5 / 2))  # ≈ 6.504 mm
VERTICAL_APERTURE   = HORIZONTAL_APERTURE * 9 / 16                               # 16:9 sensor


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
    if not rclpy.ok():
        try:
            rclpy.init()
        except Exception as e:
            raise RuntimeError("zoom_node: failed to init rclpy") from e

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

    if not camera_prim.IsValid():
        print(f"[zoom_node] WARNING: camera prim not found at '{CAMERA_PRIM_PATH}'")
        db.internal_state.focal_attr = None
    else:
        cam = UsdGeom.Camera(camera_prim)

        # Fix sensor geometry once at startup so FOV is correct at every zoom level
        cam.GetHorizontalApertureAttr().Set(float(HORIZONTAL_APERTURE))
        cam.GetVerticalApertureAttr().Set(float(VERTICAL_APERTURE))
        cam.GetFocalLengthAttr().Set(float(FOCAL_LENGTH_WIDE))   # start at wide

        db.internal_state.focal_attr = cam.GetFocalLengthAttr()
        print(f"[zoom_node] camera ready — HA={HORIZONTAL_APERTURE:.3f} mm  "
              f"FL={FOCAL_LENGTH_WIDE} → {FOCAL_LENGTH_TELE} mm")

    sub = ZoomSubscriber()
    sub.subscribe()
    db.internal_state.zoom_subscriber = sub
    db.internal_state.last_zoom = None
    print("[zoom_node] setup complete")


def compute(db):
    zoom = db.internal_state.zoom_subscriber.zoom

    if zoom is None or zoom == db.internal_state.last_zoom:
        return True
    if db.internal_state.focal_attr is None:
        return True

    zoom = max(0.0, min(1.0, zoom))
    focal_length = FOCAL_LENGTH_WIDE + zoom * (FOCAL_LENGTH_TELE - FOCAL_LENGTH_WIDE)
    db.internal_state.focal_attr.Set(float(focal_length))
    db.internal_state.last_zoom = zoom

    hfov = 2 * math.degrees(math.atan(HORIZONTAL_APERTURE / (2 * focal_length)))
    print(f"[zoom_node] zoom={zoom:.3f}  FL={focal_length:.1f} mm  HFoV={hfov:.1f}°")

    return True


def cleanup(db):
    try:
        db.internal_state.zoom_subscriber.node.destroy_node()
    except Exception:
        pass
    db.internal_state.focal_attr = None
    db.internal_state.zoom_subscriber = None
    db.internal_state.last_zoom = None
