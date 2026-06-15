"""
This file contains consts used to configure and initialize the simulation
"""

# ========================= Camera consts ============================= #
GIMBAL_ROLL_DEG = 0.0
GIMBAL_PITCH_DEG = 0.0
GIMBAL_YAW_DEG = 0.0
RESOLUTION_WIDTH = 1920     
RESOLUTION_HEIGHT = 1080         
CAMERA_FOV = 78.1                 # degrees
FOCAL_LENGTH = 22.7885            # mm, computed from FOV and sensor size


# ========================== Cesium consts ============================ #
TILESETS_HTTP_SERVER_URL = "http://10.20.15.122:8088"


# ===================== Distance sensor consts ======================== #
LASER_MIN_RANGE = 0.2
LASER_MAX_RANGE = 180.0


# ======================= ROS2 output consts ========================== #
MAX_OUTPUTS_ROS_HRZ = 30.0   

GLOBAL_POSE_TOPIC_NAME = "/isaac_core/global_pose"
LASER_TOPIC_NAME = "/isaac_core/distance_sensor"
BBOXES_TOPIC_NAME = "/isaac_core/bbox"
IMAGE_PUBLISHER_TOPIC_NAME = "/isaac_core/image_rgb"


# =========================== Image RTSP ============================== #
# Port 554 is the standard RTSP port (matches CameraConfig.rtsp_main/rtsp_alt).
# Binding < 1024 requires root or CAP_NET_BIND_SERVICE on Linux.
RTSP_PORT = 8554
# Both paths the Jetson's CameraConfig properties construct:
#   rtsp_main → rtsp://.../unicast/c1/s0/live
#   rtsp_alt  → rtsp://.../cam/realmonitor?...  (query string stripped by server)
RTSP_PATHS = ["/unicast/c1/s0/live", "/cam/realmonitor"]

# Legacy UDP constants — kept for reference, no longer used by ImageRTPStreamer
RTP_VIDEO_PORT = 5004
RTP_META_PORT = 5005

# ============================ Network ================================ #
HOST_IP = "127.0.0.1"
