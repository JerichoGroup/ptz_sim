"""UDP PTZ simulator controller — drop-in replacement for PTZController.

``PTZSimController`` mirrors the public API of
:class:`dronetracker.ptz.controller.PTZController` but, instead of driving a
real ONVIF camera, serialises each command to JSON and sends it over UDP to a
simulator host.  The simulator streams synthetic video over RTSP (point
``camera.rtsp_main_override`` at it) and sends pose updates back over UDP.

Transport / NAT note
--------------------
The Jetson reaches the simulator through a bridge that NAT-masquerades its
traffic onto a VPN, so the simulator can NEVER initiate a packet back to the
Jetson — it can only *reply* to the source address of packets it receives.
Two consequences drive the design here:

* **One socket.** Commands are sent from, and pose is received on, the SAME
  UDP socket bound to ``listen_port``.  That way the simulator's reply (sent to
  the source of our commands) traverses the NAT mapping back to this exact
  socket.  Using a separate send socket would land the reply on the wrong port.
* **Heartbeat.** A small periodic ``ping`` keeps the NAT mapping open and keeps
  the simulator's reply address fresh even when the operator isn't sending
  commands, so pose keeps flowing during idle periods.

Inject an instance into ``LivePtzPipeline(cfg, ptz=...)`` — the pipeline and its
keyboard wiring call exactly the same methods as for the real controller.
"""

__all__ = ["PTZSimController"]

import json
import socket
import threading
import time

try:
    import netz250
except ImportError:
    from ptz import netz250


class PTZSimController:
    """Jetson-side PTZ simulation controller.

    Drop-in replacement for :class:`PTZController` — same public API, UDP
    transport.
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
        heartbeat_s=1.5,
    ):
        # Networking — ONE socket bound to listen_port, used for both sending
        # commands and receiving pose (see NAT note in the module docstring).
        self.host_addr = (host_ip, host_port)
        self.listen_port = listen_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", listen_port))

        # PTZ state (mirrors real controller)
        self.zoom_search_pos = zoom_search_pos
        self.zoom_track_pos = zoom_track_pos
        self.zoom_max = zoom_max
        self.zoom_step = zoom_step
        self.cmd_ttl = cmd_ttl
        self.min_vel_scale = min_vel_scale
        self._heartbeat_s = heartbeat_s

        # Start at the wide end, like the real controller after startup-zoom,
        # so vel_scale()/_zoom_mag() are correct from the first IDLE frame.
        self.zoom_pos = zoom_search_pos
        self._zooming = False
        self._rel_moving = False

        # ContinuousMove coalescing slot
        self._cmd_vel = (0.0, 0.0)
        self._cmd_t = 0.0
        self._cmd_lock = threading.Lock()

        self.ready = True          # always ready in simulation
        self.connect_error = None  # never offline; kept for HUD API parity

        # Pose cache (populated by the listener) + background threads.
        self.pose = (0.0, 0.0, zoom_search_pos, 0.0)
        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    # ----------------------------------------------------------------------
    # Networking
    # ----------------------------------------------------------------------

    def _send(self, msg_type, **kwargs):
        packet = {"type": msg_type, "time": time.time(), **kwargs}
        self.sock.sendto(json.dumps(packet).encode("utf-8"), self.host_addr)

    def _listen_loop(self):
        """Receive pose on the same socket we send commands from."""
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                if msg.get("type") == "pose":
                    self.pose = (
                        msg["pan"],
                        msg["tilt"],
                        msg["zoom"],
                        msg["timestamp"],
                    )
                    # Sync zoom_pos to simulator truth, mirroring the real
                    # controller's _poll_pose (otherwise zoom_pos goes stale
                    # after center_and_zoom, which never updates it locally).
                    # Pose carries RAW ONVIF zoom (Netz-250: -1..+1); normalise
                    # to [0,1] like the real controller does.
                    self.zoom_pos = netz250.normalize_zoom(msg["zoom"])
            except Exception:
                pass

    def _heartbeat_loop(self):
        """Periodic ping so the NAT mapping stays open and the simulator keeps a
        fresh reply address even when no commands are being sent."""
        while True:
            try:
                self._send("ping")
            except Exception:
                pass
            time.sleep(self._heartbeat_s)

    # ----------------------------------------------------------------------
    # Public API (mirrors real PTZController)
    # ----------------------------------------------------------------------

    def get_pose(self):
        return self.pose

    # --- Zoom --------------------------------------------------------------

    def zoom_to(self, t):
        t = float(max(0.0, min(1.0, t)))
        self.zoom_pos = t
        # The host is a Netz-250 emulator: it expects the RAW ONVIF zoom value,
        # which is what the camera's ONVIF interface would receive.
        self._send("zoom_to", raw=netz250.denormalize_zoom(t))

    def zoom_in(self):
        self.zoom_to(min(self.zoom_max, self.zoom_pos + self.zoom_step))

    def zoom_out(self):
        self.zoom_to(max(0.0, self.zoom_pos - self.zoom_step))

    def zoom_search(self):
        self.zoom_to(self.zoom_search_pos)

    def zoom_track(self):
        self.zoom_to(self.zoom_track_pos)

    # --- Velocity scale (matches real controller) ---------------------------

    def vel_scale(self) -> float:
        """Pan/tilt velocity scale factor at the current zoom level.

        Returns 1.0 at zoom_pos=0 (widest FOV), decreasing linearly toward
        min_vel_scale at zoom_pos=1 (max zoom). Matches the real
        PTZController.vel_scale() formula exactly (see controller.py).
        """
        return max(self.min_vel_scale, 1.0 - self.zoom_pos * (1.0 - self.min_vel_scale))

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
