#!/usr/bin/env python3
import socket
import json
import threading
import time

from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
import isaac_core_dev_kit.dev_utils as core_utils


class PTZSim:
    """
    Host-side PTZ simulation controller.

    - Receives PTZ commands over UDP from Jetson (PTZSimController).
    - Uses Isaac Sim via HostIsaacManager.
    - Uses dev_utils.set_gimbal_angle() for pan/tilt.
    - Maintains internal zoom state (self._zoom).
    """

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
        self._pan = 0.0    # yaw
        self._tilt = 0.0   # pitch
        self._roll = 0.0   # keep roll at 0 for now
        self._zoom = 1.0   # [0, 1] normalized zoom

        # Networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", listen_port))

        # Pose sender socket
        self.pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Control flags
        self._run = True

        # Start threads
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._pose_loop, daemon=True).start()

        print(f"[Host] PTZSim listening on 0.0.0.0:{listen_port}")
        print(f"[Host] Jetson pose target: {self.jetson_addr}")

        # Isaac Sim setup
        ## core_utils.delete_cesium_cache()
        core_utils.safe_rclpy_init()

        self._isaac_ctx = HostIsaacManager(
            core_path=core_path,
            usd_path=usd_path,
            com_udp=True,
            show_isaac_logs=show_isaac_logs,
        )

    # ----------------------------------------------------------------------
    # Context manager to run Isaac Sim
    # ----------------------------------------------------------------------

    def run(self):
        """
        Run PTZSim inside HostIsaacManager context.
        Blocks until interrupted.
        """
        with self._isaac_ctx:
            print("[Host] Isaac Sim started (HostIsaacManager).")
            try:
                while self._run:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("[Host] KeyboardInterrupt, shutting down PTZSim.")
            finally:
                self._run = False
                core_utils.safe_rclpy_shutdown()

    # ----------------------------------------------------------------------
    # Receive commands from Jetson
    # ----------------------------------------------------------------------

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
            # For now just log; you can add real home logic later
            print("[Host] save_home (not implemented yet)")

        elif t == "go_home":
            print("[Host] go_home (not implemented yet)")

        elif t == "stop_all":
            print("[Host] stop_all")
            self._run = False

    # ----------------------------------------------------------------------
    # Simulated PTZ logic using dev_utils.set_gimbal_angle
    # ----------------------------------------------------------------------

    def _apply_move(self, vx, vy):
        """
        Continuous-like move: small increments to pan/tilt.
        vx, vy are velocities in normalized units.
        """
        # Scale factors are arbitrary for now; tune later
        self._pan += float(vx) * 5.0
        self._tilt += float(vy) * 5.0
        self._update_gimbal()

    def _apply_relative(self, dp, dt):
        """
        Relative move: direct offsets to pan/tilt.
        """
        self._pan += float(dp) * 180.0   # map ONVIF-ish units to degrees
        self._tilt += float(dt) * 180.0
        self._update_gimbal()

    def _apply_zoom(self, target):
        """
        Zoom: just update internal zoom state for now.
        """
        self._zoom = float(max(0.0, min(1.0, target)))
        print(f"[Host] Zoom set to {self._zoom:.2f} (normalized)")

    def _update_gimbal(self):
        """
        Apply current pan/tilt/roll to Isaac Sim via dev_utils.
        """
        print(f"[Host] set_gimbal_angle(roll={self._roll:.2f}, "
              f"pitch={self._tilt:.2f}, yaw={self._pan:.2f})")
        try:
            core_utils.set_gimbal_angle(
                roll=self._roll,
                pitch=self._tilt,
                yaw=self._pan,
            )
        except Exception as e:
            print(f"[Host] set_gimbal_angle error: {e}")

    # ----------------------------------------------------------------------
    # Send pose updates back to Jetson
    # ----------------------------------------------------------------------

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
            time.sleep(0.05)  # 20 Hz pose updates


if __name__ == "__main__":
    # Adjust these to your environment
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
