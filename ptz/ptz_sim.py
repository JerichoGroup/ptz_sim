#!/usr/bin/env python3
import socket
import json
import threading
import time
import math

from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.one_point_sender import OnePointSender
import isaac_core_dev_kit.dev_utils as core_utils


class PTZSim:
    """
    Host-side PTZ simulation controller.

    - Receives PTZ commands over UDP from Jetson (PTZSimController).
    - Uses Isaac Sim via HostIsaacManager.
    - Uses dev_utils.set_gimbal_angle() for pan/tilt.
    - Maintains internal zoom state (self._zoom).
    - Implements save_home() and go_home() with smooth slewing.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        listen_port=5005,
        jetson_ip=None,
        jetson_port=5006,
        core_path="/home/ofer/clones/ptz_sim",
        usd_path="./usd/maps/earth/earth.usda",
        show_isaac_logs=False,
    ):
        self.listen_port = listen_port
        self.jetson_addr = (jetson_ip, jetson_port)

        # PTZ state
        self._pan = 0.0     # yaw (degrees)
        self._tilt = 0.0    # pitch (degrees)
        self._roll = 0.0    # roll (degrees)
        self._zoom = 1.0    # normalized [0..1]

        # Home state
        self._home_pan = None
        self._home_tilt = None
        self._home_zoom = None

        # Networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", listen_port))

        self.pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Control flags
        self._run = True
        self._slewing_home = False

        # Start threads
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._pose_loop, daemon=True).start()

        print(f"[Host] PTZSim listening on 0.0.0.0:{listen_port}")
        print(f"[Host] Jetson pose target: {self.jetson_addr}")

        # Isaac Sim setup
        core_utils.delete_cesium_cache()
        core_utils.safe_rclpy_init()

        self._isaac_ctx = HostIsaacManager(
            core_path=core_path,
            usd_path=usd_path,
            com_udp=True,
            show_isaac_logs=show_isaac_logs,
        )

    # ------------------------------------------------------------------
    # Context manager wrapper
    # ------------------------------------------------------------------

    def run(self):
        """Run PTZSim inside HostIsaacManager context."""
        with self._isaac_ctx:
            print("[Host] Isaac Sim started.")
            try:
                self._init_camera_position()
                while self._run:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("[Host] KeyboardInterrupt, shutting down.")
            finally:
                self._run = False
                core_utils.safe_rclpy_shutdown()

    # ------------------------------------------------------------------
    # UDP receive loop
    # ------------------------------------------------------------------

    def _recv_loop(self):
        while self._run:
            try:
                data, addr = self.sock.recvfrom(4096)
            except OSError:
                break

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            self._handle_cmd(msg, addr)

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    def _handle_cmd(self, msg, addr):
        print(f"[Host] From {addr}: {msg}")

        t = msg.get("type")

        if t == "move":
            self._apply_move(msg["vx"], msg["vy"])

        elif t == "relative_move":
            self._apply_relative(msg["dp"], msg["dt"])

        elif t == "zoom_to":
            self._apply_zoom(msg["target"])

        elif t == "center_and_zoom":
            self._apply_relative(msg["dp"], msg["dt"])
            self._apply_zoom(msg["zoom_target"])

        elif t == "save_home":
            self._save_home()

        elif t == "go_home":
            self._go_home()

        elif t == "stop_all":
            print("[Host] stop_all")
            self._run = False

    # ------------------------------------------------------------------
    # PTZ motion logic
    # ------------------------------------------------------------------

    def _apply_move(self, vx, vy):
        """
        ContinuousMove-like behavior.
        vx, vy are normalized velocities [-1..1].
        """
        # Scale to degrees per command tick
        self._pan += float(vx) * 2.0
        self._tilt += float(vy) * 2.0
        self._update_gimbal()

    def _apply_relative(self, dp, dt):
        """
        RelativeMove-like behavior.
        dp, dt are normalized deltas [-1..1].
        """
        # Scale to degrees
        self._pan += float(dp) * 45.0
        self._tilt += float(dt) * 45.0
        self._update_gimbal()

    def _apply_zoom(self, target):
        """Update zoom state."""
        self._zoom = float(max(0.0, min(1.0, target)))
        print(f"[Host] Zoom set to {self._zoom:.2f}")

    # ------------------------------------------------------------------
    # Home logic (save + smooth slew)
    # ------------------------------------------------------------------

    def _save_home(self):
        self._home_pan = self._pan
        self._home_tilt = self._tilt
        self._home_zoom = self._zoom
        print(f"[Host] Home saved: pan={self._home_pan:.2f}, tilt={self._home_tilt:.2f}, zoom={self._home_zoom:.2f}")

    def _go_home(self):
        if self._home_pan is None:
            print("[Host] go_home called but no home saved.")
            return

        if self._slewing_home:
            print("[Host] Already slewing home.")
            return

        print("[Host] Slewing back to home position...")
        threading.Thread(target=self._slew_home_thread, daemon=True).start()

    def _slew_home_thread(self):
        self._slewing_home = True

        start_pan = self._pan
        start_tilt = self._tilt
        start_zoom = self._zoom

        end_pan = self._home_pan
        end_tilt = self._home_tilt
        end_zoom = self._home_zoom

        duration = 2.0  # seconds
        steps = int(duration / 0.02)

        for i in range(steps):
            if not self._run:
                break

            alpha = i / steps

            # Smooth interpolation (ease-in-out)
            s = 0.5 - 0.5 * math.cos(math.pi * alpha)

            self._pan = start_pan + (end_pan - start_pan) * s
            self._tilt = start_tilt + (end_tilt - start_tilt) * s
            self._zoom = start_zoom + (end_zoom - start_zoom) * s

            self._update_gimbal()
            time.sleep(0.02)

        print("[Host] Home reached.")
        self._slewing_home = False

    # ------------------------------------------------------------------
    # Apply gimbal angles to Isaac Sim
    # ------------------------------------------------------------------

    def _update_gimbal(self):
        """Send roll/pitch/yaw to Isaac Sim."""
        try:
            core_utils.set_gimbal_angle(
                roll=self._roll,
                pitch=self._tilt,
                yaw=self._pan,
            )
        except Exception as e:
            print(f"[Host] set_gimbal_angle error: {e}")

    # ------------------------------------------------------------------
    # Move Camera to initial position
    # ------------------------------------------------------------------
    def _init_camera_position(self):

        camera_point_sender = OnePointSender(lat=32.19965, lon=35.30593, alt=1000.0,
                            roll=0.0, pitch=0.0, yaw=0.0)
        
        camera_point_sender.send_once(32.19965, 35.30593, 1000.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Pose updates back to Jetson
    # ------------------------------------------------------------------

    def _pose_loop(self):
        while self._run:
            if self.jetson_addr[0] is not None:
                packet = {
                    "type": "pose",
                    "pan": self._pan,
                    "tilt": self._tilt,
                    "zoom": self._zoom,
                    "timestamp": time.time(),
                }
                try:
                    self.pose_sock.sendto(
                        json.dumps(packet).encode("utf-8"),
                        self.jetson_addr,
                    )
                except Exception:
                    pass

            time.sleep(0.05)  # 20 Hz


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    CORE_PATH = "/home/ofer/clones/ptz_sim"
    USD_PATH = "./usd/maps/earth/earth.usda"

    JETSON_IP = "192.168.55.1"
    JETSON_PORT = 5006

    sim = PTZSim(
        listen_port=5005,
        jetson_ip=JETSON_IP,
        jetson_port=JETSON_PORT,
        core_path=CORE_PATH,
        usd_path=USD_PATH,
        show_isaac_logs=False,
    )
    sim.run()
