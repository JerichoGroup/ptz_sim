import math
import socket
import threading
import time
import json

try:
    from dronetracker.ptz.transform import _F_WIDE, _F_TELE, _SENSOR_W
except ImportError:
    # Fallback constants matching the UNV IPC6852ER-X45 (45x optical zoom)
    _F_WIDE = 4.56      # mm, wide-end focal length
    _F_TELE = 205.2     # mm, tele-end focal length  (~45x)
    _SENSOR_W = 5.76    # mm, sensor width → FOV_WIDE ≈ 64°

_FOV_WIDE = 2 * math.degrees(math.atan(_SENSOR_W / (2 * _F_WIDE)))


class PTZSimController:
    """
    Jetson-side PTZ simulation controller.
    Drop-in replacement for PTZController — same public API, UDP transport.
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
        min_vel_scale=0.30,
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
        self.min_vel_scale = min_vel_scale

        self.zoom_pos = 1.0
        self._zooming = False
        self._rel_moving = False

        # ContinuousMove coalescing slot
        self._cmd_vel = (0.0, 0.0)
        self._cmd_t = 0.0
        self._cmd_lock = threading.Lock()

        self.ready = True  # always ready in simulation

    # ----------------------------------------------------------------------
    # Networking
    # ----------------------------------------------------------------------

    def _send(self, msg_type, **kwargs):
        packet = {"type": msg_type, "time": time.time(), **kwargs}
        self.sock.sendto(json.dumps(packet).encode("utf-8"), self.host_addr)

    def _listen_loop(self):
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

    # --- Velocity scale (FOV-relative, matches real controller) ------------

    def vel_scale(self) -> float:
        """Pan/tilt velocity scale factor at the current zoom level.

        Returns 1.0 at widest FOV, decreasing toward min_vel_scale at max zoom.
        Matches the real PTZController.vel_scale() formula exactly.
        """
        f = self.zoom_pos * (_F_TELE - _F_WIDE) + _F_WIDE
        fov = 2 * math.degrees(math.atan(_SENSOR_W / (2 * f)))
        return max(self.min_vel_scale, fov / _FOV_WIDE)

    # --- Autofocus (no-op in simulation) -----------------------------------

    def trigger_autofocus(self):
        pass

    # --- Pan/Tilt ContinuousMove ------------------------------------------

    def move(self, vx, vy):
        """Write latest-intent velocity to coalescing slot and send to host.

        Returns early if a RelativeMove is in flight, matching real controller
        behaviour (ContinuousMove would cancel the in-flight RelativeMove).
        """
        if self._rel_moving:
            return
        with self._cmd_lock:
            self._cmd_vel = (float(vx), float(vy))
            self._cmd_t = time.time()
        self._send("move", vx=vx, vy=vy)

    pid_move = move

    # --- RelativeMove ------------------------------------------------------

    def relative_move(self, dp, dt, space=None):
        """One-shot relative pan/tilt; gated so rapid calls don't flood the host."""
        if self._zooming or self._rel_moving:
            return
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        self._rel_moving = True
        self._send("relative_move", dp=dp, dt=dt, space=space)
        threading.Thread(target=self._settle, args=(0.5, "_rel_moving"), daemon=True).start()

    # --- Center and Zoom ---------------------------------------------------

    def center_and_zoom(self, dp, dt, zoom_target, space=None, parallel=True):
        """RelativeMove + zoom_to in one command; both flags held for settle duration."""
        if self._zooming or self._rel_moving:
            return False
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        self._rel_moving = True
        self._zooming = True
        self._send(
            "center_and_zoom",
            dp=dp, dt=dt, zoom_target=zoom_target,
            space=space, parallel=parallel,
        )
        threading.Thread(target=self._settle_both, args=(0.6,), daemon=True).start()
        return True

    # --- Home --------------------------------------------------------------

    def save_home(self):
        self._send("save_home")

    def go_home(self):
        self._send("go_home")

    # --- Shutdown ----------------------------------------------------------

    def stop_all(self):
        self._send("stop_all")

    # ----------------------------------------------------------------------
    # Settle helpers — clear motion flags after simulated settle time
    # ----------------------------------------------------------------------

    def _settle(self, delay, flag):
        time.sleep(delay)
        setattr(self, flag, False)

    def _settle_both(self, delay):
        time.sleep(delay)
        self._rel_moving = False
        self._zooming = False
