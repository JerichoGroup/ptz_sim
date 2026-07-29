"""
This file contains consts used to configure and initialize the simulation
"""

# ========================= Camera consts ============================= #
# Netz-250 (Sony IMX415 1/2.8", 20x optical).
GIMBAL_ROLL_DEG = 0.0
GIMBAL_PITCH_DEG = 0.0
GIMBAL_YAW_DEG = 0.0
RESOLUTION_WIDTH = 1920
RESOLUTION_HEIGHT = 1080
# Wide-end optics. Derived from the measured pan-shift curve, NOT the datasheet
# (see ptz/netz250.py for why). zoom_node.py overrides these per zoom level;
# these are just the startup state.
CAMERA_FOV = 64.98                # degrees, horizontal, at the wide end
FOCAL_LENGTH = 4.3731             # mm, gives HFoV=64.98 with a 5.57 mm sensor
CAMERA_FPS = 20.0                 # Netz-250 native stream rate


# ========================== Cesium consts ============================ #
TILESETS_HTTP_SERVER_URL = "http://192.168.30.71:8088"


# ===================== Distance sensor consts ======================== #
LASER_MIN_RANGE = 0.2
LASER_MAX_RANGE = 180.0


# ======================= ROS2 output consts ========================== #
MAX_OUTPUTS_ROS_HRZ = CAMERA_FPS   # match the camera's native stream rate

GLOBAL_POSE_TOPIC_NAME = "/isaac_core/global_pose"
LASER_TOPIC_NAME = "/isaac_core/distance_sensor"
BBOXES_TOPIC_NAME = "/isaac_core/bbox"
IMAGE_PUBLISHER_TOPIC_NAME = "/isaac_core/image_rgb"


# =========================== Image RTSP ============================== #
# Port 554 is the standard RTSP port (matches CameraConfig.rtsp_main/rtsp_alt).
# Binding < 1024 requires root or CAP_NET_BIND_SERVICE on Linux.
RTSP_PORT = 8554
# Both paths the Jetson's CameraConfig properties construct.  The Netz-250 uses
# a Hikvision-style main path; the /cam/realmonitor alt path is kept because
# DroneTracker's RtspConfig still falls back to it.
#   rtsp_main → rtsp://.../Streaming/channel/1
#   rtsp_alt  → rtsp://.../cam/realmonitor?...  (query string stripped by server)
# The legacy UNV path stays mounted so the older tools/URLs keep working.
RTSP_PATHS = [
    "/Streaming/channel/1",
    "/cam/realmonitor",
    "/unicast/c1/s0/live",
]

# Legacy UDP constants — kept for reference, no longer used by ImageRTPStreamer
RTP_VIDEO_PORT = 5004
RTP_META_PORT = 5005

# ============================ Network ================================ #
HOST_IP = "127.0.0.1"
