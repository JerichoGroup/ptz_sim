import socket
import threading
import time
import json


class PTZSimController:
    """
    Jetson-side PTZ simulation controller.
    Drop-in replacement for PTZController.
    Sends commands to the Host over UDP.
    """

    def __init__(
        self,
        host_ip,
        host_port=5005,
        listen_port=5006,
        zoom_search_pos=0.0,
        zoom_track_pos=0.60,
        zoom_max=1.0,
        zoom_step=0.05,
        cmd_ttl=0.15,
    ):
        # Networking
        self.host_addr = (host_ip, host_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Listener for pose updates
        self.listen_port = listen_port
        self.pose = (0.0, 0.0, 0.0, 0.0)
        threading.Thread(target=self._listen_loop, daemon=True).start()

        # PTZ state (mirrors real controller)
        self.zoom_search_pos = zoom_search_pos
        self.zoom_track_pos = zoom_track_pos
        self.zoom_max = zoom_max
        self.zoom_step = zoom_step
        self.cmd_ttl = cmd_ttl

        self.zoom_pos = 1.0
        self._zooming = False
        self._rel_moving = False
        self._cmd_vel = (0.0, 0.0)
        self._cmd_t = 0.0

        self.ready = True  # Always ready in simulation

    # ----------------------------------------------------------------------
    # Networking
    # ----------------------------------------------------------------------

    def _send(self, msg_type, **kwargs):
        """Send a JSON command packet to the host."""
        packet = {"type": msg_type, "time": time.time(), **kwargs}
        data = json.dumps(packet).encode("utf-8")
        self.sock.sendto(data, self.host_addr)

    def _listen_loop(self):
        """Listen for pose updates from the host."""
        lsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("0.0.0.0", self.listen_port))

        while True:
            try:
                data, _ = lsock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                if msg.get("type") == "pose":
                    self.pose = (
                        msg["pan"],
                        msg["tilt"],
                        msg["zoom"],
                        msg["timestamp"],
                    )
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # Public API (mirrors real PTZController)
    # ----------------------------------------------------------------------

    def get_pose(self):
        return self.pose

    # --- Zoom --------------------------------------------------------------

    def zoom_to(self, t):
        t = float(max(0.0, min(1.0, t)))
        self.zoom_pos = t
        self._send("zoom_to", target=t)

    def zoom_in(self):
        self.zoom_to(min(self.zoom_max, self.zoom_pos + self.zoom_step))

    def zoom_out(self):
        self.zoom_to(max(0.0, self.zoom_pos - self.zoom_step))

    def zoom_search(self):
        self.zoom_to(self.zoom_search_pos)

    def zoom_track(self):
        self.zoom_to(self.zoom_track_pos)

    # --- Pan/Tilt ContinuousMove ------------------------------------------

    def move(self, vx, vy):
        self._cmd_vel = (float(vx), float(vy))
        self._cmd_t = time.time()
        self._send("move", vx=vx, vy=vy)

    pid_move = move

    # --- RelativeMove ------------------------------------------------------

    def relative_move(self, dp, dt, space=None):
        self._send("relative_move", dp=dp, dt=dt, space=space)

    # --- Center and Zoom ---------------------------------------------------

    def center_and_zoom(self, dp, dt, zoom_target, space=None, parallel=True):
        self._send(
            "center_and_zoom",
            dp=dp,
            dt=dt,
            zoom_target=zoom_target,
            space=space,
            parallel=parallel,
        )

    # --- Home --------------------------------------------------------------

    def save_home(self):
        self._send("save_home")

    def go_home(self):
        self._send("go_home")

    # --- Shutdown ----------------------------------------------------------

    def stop_all(self):
        self._send("stop_all")
