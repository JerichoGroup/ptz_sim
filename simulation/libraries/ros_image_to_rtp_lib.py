"""ROS2 Image → RTSP streaming library.

Subscribes to a ROS2 Image topic, encodes frames as H.264 via GStreamer,
and serves them over RTSP using GstRtspServer.

The server mounts the stream at every path in rtsp_paths so that both URLs
the Jetson's CameraConfig constructs work out of the box:
  rtsp_main → rtsp://<ip>:554/unicast/c1/s0/live
  rtsp_alt  → rtsp://<ip>:554/cam/realmonitor?...  (query string is ignored
              by GstRtspServer's mount-point lookup, path component only)

NOTE: binding port 554 requires root or CAP_NET_BIND_SERVICE on Linux.
      Run with sudo, or forward with:
          sudo iptables -t nat -A PREROUTING -p tcp --dport 554 -j REDIRECT --to-port 8554
      and change RTSP_PORT in consts.py to 8554.
"""

import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GstRtspServer

from libraries.sim_lib import SimLibBase


# ==================== GstRTSPBridge ====================

class GstRTSPBridge:
    """Encodes ROS2 Image messages with GStreamer and serves them as RTSP.

    Lifecycle:
      - No client connected → frames are dropped (appsrc is None).
      - First client connects → GstRtspServer fires media-configure → appsrc
        reference is saved → subsequent frames are pushed into the pipeline.
      - Last client disconnects → media-unprepared → appsrc reference cleared.
      - Next client reconnects → media-configure fires again with a fresh pipeline.
    """

    # pay0 is the mandatory payloader name required by GstRtspServer.
    _FACTORY_PIPELINE = (
        "( appsrc name=src is-live=true do-timestamp=true block=false format=time "
        "! videoconvert "
        "! x264enc tune=zerolatency bitrate=10000 speed-preset=superfast "
        "! h264parse "
        "! rtph264pay config-interval=1 pt=96 name=pay0 )"
    )

    def __init__(self, rtsp_port: int, rtsp_paths: list) -> None:
        self._rtsp_port = rtsp_port
        self._rtsp_paths = rtsp_paths

        # appsrc reference — set by media-configure, cleared by media-unprepared.
        # Guarded by _appsrc_lock because media-configure fires in the GLib thread
        # while send_image runs in the ROS spin thread.
        self._appsrc = None
        self._appsrc_lock = threading.Lock()
        self._last_caps_str = None

        Gst.init(None)
        self._create_rtsp_server()

        # GLib main loop drives the RTSP server event loop.
        self._main_loop = GObject.MainLoop()
        self._gst_thread = threading.Thread(target=self._main_loop.run, daemon=True)
        self._gst_thread.start()

        print(f"[RTSP] Server listening on :{rtsp_port}  paths: {rtsp_paths}")
        print(f"[RTSP] Connect Jetson with camera.ip = <host-ip> in config.yaml")

    def _create_rtsp_server(self) -> None:
        server = GstRtspServer.RTSPServer.new()
        server.props.service = str(self._rtsp_port)

        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(self._FACTORY_PIPELINE)
        factory.set_shared(True)  # all clients share one pipeline instance
        factory.connect("media-configure", self._on_media_configure)

        mounts = server.get_mount_points()
        for path in self._rtsp_paths:
            mounts.add_factory(path, factory)
            print(f"[RTSP] Mounted at {path}")

        server.attach(None)  # attaches to default GLib main context
        self._rtsp_server = server

    def _on_media_configure(self, factory, media) -> None:
        """Fired in GLib thread when the first RTSP client connects."""
        pipeline = media.get_element()
        appsrc = pipeline.get_by_name("src")
        if appsrc is None:
            print("[RTSP] WARNING: appsrc 'src' not found in factory pipeline")
            return

        # Restore caps if we already know the frame format from a previous session.
        if self._last_caps_str:
            appsrc.set_property("caps", Gst.Caps.from_string(self._last_caps_str))

        with self._appsrc_lock:
            self._appsrc = appsrc

        media.connect("unprepared", self._on_media_unprepared)
        print("[RTSP] Client connected — pipeline started")

    def _on_media_unprepared(self, media) -> None:
        """Fired in GLib thread when the last RTSP client disconnects."""
        with self._appsrc_lock:
            self._appsrc = None
        # Reset caps so they are re-applied cleanly on the next connection.
        self._last_caps_str = None
        print("[RTSP] All clients disconnected — pipeline stopped")

    @staticmethod
    def _caps_for(encoding: str, width: int, height: int):
        fmt_map = {
            "rgb8":  "RGB",
            "bgr8":  "BGR",
            "mono8": "GRAY8",
            "rgba8": "RGBA",
            "bgra8": "BGRA",
        }
        fmt = fmt_map.get(encoding)
        if fmt is None:
            return None
        return Gst.Caps.from_string(
            f"video/x-raw,format={fmt},width={width},height={height},framerate=30/1"
        )

    def send_image(self, msg: Image) -> None:
        """Push one ROS Image frame into the RTSP pipeline."""
        with self._appsrc_lock:
            appsrc = self._appsrc
        if appsrc is None:
            return  # no client connected — drop frame silently

        try:
            raw = bytes(msg.data)
        except Exception as e:
            print("[RTSP Bridge] Failed to copy image data:", e)
            return

        caps = self._caps_for(msg.encoding, msg.width, msg.height)
        if caps is None:
            print(f"[RTSP Bridge] Unsupported encoding: {msg.encoding}")
            return

        caps_str = caps.to_string()
        if caps_str != self._last_caps_str:
            appsrc.set_property("caps", caps)
            self._last_caps_str = caps_str

        buf = Gst.Buffer.new_allocate(None, len(raw), None)
        buf.fill(0, raw)
        appsrc.emit("push-buffer", buf)

    def close(self) -> None:
        with self._appsrc_lock:
            self._appsrc = None
        if self._main_loop:
            try:
                self._main_loop.quit()
            except Exception:
                pass
        if self._gst_thread:
            self._gst_thread.join(timeout=0.5)


# ==================== ROS2 node ====================

class RosImageToRTSPNode(Node):
    """ROS2 node: subscribes to an Image topic and feeds frames to GstRTSPBridge."""

    def __init__(self, topic: str, rtsp_port: int, rtsp_paths: list) -> None:
        super().__init__("ros_image_to_rtsp")
        self.bridge = GstRTSPBridge(rtsp_port, rtsp_paths)
        self.sub = self.create_subscription(Image, topic, self._callback, 10)
        self.get_logger().info(
            f"Subscribing to {topic}, RTSP server on :{rtsp_port}"
        )

    def _callback(self, msg: Image) -> None:
        self.bridge.send_image(msg)

    def destroy_node(self) -> None:
        self.bridge.close()
        super().destroy_node()


# ==================== SimLibBase adapter ====================

class ImageRTPStreamer(SimLibBase):
    """Simulation library: streams camera images over RTSP.

    Name kept as ImageRTPStreamer for backward compatibility with lib_manager.py.
    """

    def __init__(self, topic: str, rtsp_port: int, rtsp_paths: list) -> None:
        self.topic = topic
        self.rtsp_port = rtsp_port
        self.rtsp_paths = rtsp_paths
        self.node = None

    def start(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = RosImageToRTSPNode(self.topic, self.rtsp_port, self.rtsp_paths)
        try:
            rclpy.spin(self.node)
        except Exception as e:
            print("[ImageRTPStreamer] Exception in rclpy.spin:", e)

    def shutdown(self) -> None:
        if self.node:
            self.node.destroy_node()
            self.node = None
        if rclpy.ok():
            rclpy.shutdown()
