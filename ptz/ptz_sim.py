#!/usr/bin/env python3
import socket
import json
import threading
import time
import math

import rclpy
from isaac_ros2_messages.msg import Gimbal
from std_msgs.msg import Float32

from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.one_point_sender import OnePointSender
import isaac_core_dev_kit.dev_utils as core_utils

# Scene library (sibling module). Run as `python3 ./ptz/ptz_sim.py` puts ptz/ on
# sys.path[0]; fall back to the package path if imported differently.
try:
    from scenes import run_scene
except ImportError:
    from ptz.scenes import run_scene


# Camera lens constants (UNV IPC6852ER-X45-VF) — kept identical to
# simulation/script_nodes/zoom_node.py so the angular response of relative moves
# matches the rendered FOV at every zoom level.
FOCAL_LENGTH_WIDE = 5.7      # mm
FOCAL_LENGTH_TELE = 256.5    # mm
HORIZONTAL_APERTURE = 2 * FOCAL_LENGTH_WIDE * math.tan(math.radians(59.5 / 2))  # ≈ 6.504 mm


class PTZSim:
    """
    Host-side PTZ simulation controller.

    - Receives PTZ commands over UDP from Jetson (PTZSimController).
    - Uses Isaac Sim via HostIsaacManager.
    - ContinuousMove: 25 Hz integrator loop with TTL coalescing.
    - Gimbal control via a persistent ROS2 publisher (no per-call node churn).
    - Maintains internal zoom state (self._zoom); FOV not yet wired.
    - Implements save_home() and go_home() with smooth cosine slewing.
    """

    def __init__(
        self,
        listen_port=5005,
        jetson_ip=None,
        jetson_port=5006,
        core_path="/home/ofer/clones/ptz_sim",
        usd_path="./usd/maps/earth/earth.usda",
        show_isaac_logs=False,
        image_rtp=False,
        pan_vel=60.0,   # deg/s at vx=1.0
        tilt_vel=60.0,  # deg/s at vy=1.0
        cmd_ttl=0.15,   # seconds before a coalesced move command goes stale
        scene_start_delay_s=20.0,  # wait this long after Isaac loads before playing the scene
    ):
        self.listen_port = listen_port
        # Optional explicit pose target for a directly-routable setup. Default
        # (jetson_ip=None) → reply-to-sender: the sim cannot initiate to the
        # Jetson (it sits behind a NAT/VPN bridge; 192.168.55.1 is unreachable
        # here), so pose is sent back to the source address of received packets.
        self.jetson_addr = (jetson_ip, jetson_port)
        self._last_cmd_addr = None
        self._addr_lock = threading.Lock()

        # PTZ state — protected by _state_lock for R/M/W operations
        self._pan = 0.0
        self._tilt = 0.0
        self._roll = 0.0
        self._zoom = 0.0          # current published zoom (0 = wide)
        self._zoom_target = 0.0   # target set by _apply_zoom; loop slews toward it
        self._state_lock = threading.Lock()

        # Home state
        self._home_pan = None
        self._home_tilt = None
        self._home_zoom = None

        # ContinuousMove coalescing slot
        self._pan_vel = pan_vel
        self._tilt_vel = tilt_vel
        self._cmd_ttl = cmd_ttl
        self._cmd_vel = (0.0, 0.0)
        self._cmd_t = 0.0
        self._cmd_lock = threading.Lock()

        # Control flags
        self._run = True
        self._slewing_home = False

        # Scene playback — the scene NUMBER and a per-Jetson-session token come
        # from the command channel; the START DELAY is configured here. The scene
        # is (re)played once per NEW session, so quitting the Jetson, changing
        # sim.scene, and re-running replays the scene.
        self._scene_start_delay_s = scene_start_delay_s
        self._requested_scene = None   # set from received packets; None = not told yet
        self._session_id = None        # latest Jetson session token (from the heartbeat)

        # Isaac/ROS setup — must happen before threads start so _update_gimbal is safe
        core_utils.delete_cesium_cache()
        core_utils.safe_rclpy_init()
        self._gimbal_node = rclpy.create_node("ptz_sim_gimbal_node")
        self._gimbal_pub = self._gimbal_node.create_publisher(Gimbal, "/isaac_core/gimbal", 10)
        self._zoom_pub   = self._gimbal_node.create_publisher(Float32, "/isaac_core/zoom", 10)

        # Networking — ONE socket for both receiving commands and sending pose.
        # Pose MUST go out from this socket (the port the Jetson addressed) so the
        # reply traverses the bridge's NAT mapping back to the Jetson; a separate
        # ephemeral socket would not match conntrack and would be dropped.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", listen_port))

        # Start threads
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._pose_loop, daemon=True).start()
        threading.Thread(target=self._mover_loop, daemon=True).start()
        threading.Thread(target=self._zoom_loop, daemon=True).start()

        print(f"[Host] PTZSim listening on 0.0.0.0:{listen_port}")
        if self.jetson_addr[0] is not None:
            print(f"[Host] Jetson pose target (explicit): {self.jetson_addr}")
        else:
            print("[Host] Jetson pose target: reply-to-sender (source of received commands)")

        self._isaac_ctx = HostIsaacManager(
            core_path=core_path,
            usd_path=usd_path,
            com_udp=True,
            show_isaac_logs=show_isaac_logs,
            image_rtp=image_rtp,
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
                # Auto-capture home at the startup pose (0/0/0), mirroring the
                # real controller's connect-time home capture — so the Jetson's
                # automatic go_home (R-key, TRACK→IDLE reset) works from the start.
                self._save_home()
                # Isaac is loaded now → start the scene-playback timer.
                threading.Thread(target=self._scene_loop, daemon=True).start()
                while self._run:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("[Host] KeyboardInterrupt, shutting down.")
            finally:
                self._run = False
                self._gimbal_node.destroy_node()
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
            # Remember where to reply with pose — on EVERY packet (incl. the
            # heartbeat ping), before parsing, so the reply address stays fresh.
            with self._addr_lock:
                self._last_cmd_addr = addr
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            # The Jetson carries the requested scene on the command channel
            # (typically on the heartbeat ping). Capture it from any packet that
            # has it, before the ping early-return in _handle_cmd.
            if isinstance(msg, dict):
                if msg.get("scene") is not None:
                    try:
                        self._requested_scene = int(msg["scene"])
                    except (TypeError, ValueError):
                        pass   # ignore a malformed scene value; don't kill the recv loop
                if msg.get("session") is not None:
                    self._session_id = msg["session"]
            self._handle_cmd(msg, addr)

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    def _handle_cmd(self, msg, addr):
        t = msg.get("type")
        if t == "ping":
            return   # heartbeat: only refreshes the reply address (done in _recv_loop)
        print(f"[Host] From {addr}: {msg}")

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
            self._stop_motion()

    # ------------------------------------------------------------------
    # PTZ motion logic
    # ------------------------------------------------------------------

    def _apply_move(self, vx, vy):
        """Write latest-intent velocity to the coalescing slot."""
        with self._cmd_lock:
            self._cmd_vel = (float(vx), float(vy))
            self._cmd_t = time.time()

    def _apply_relative(self, dp, dt):
        """One-shot delta move in ONVIF TranslationSpaceFov units: dp/dt are
        fractions of the *current* field of view, where ±1.0 == half the current
        HFoV of angular motion. Scaling is zoom-dependent (HFoV shrinks as we zoom
        in), unlike the old constant 45°. Clears the coalescing slot so the mover
        doesn't fight it."""
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        with self._state_lock:
            fl = FOCAL_LENGTH_WIDE + self._zoom * (FOCAL_LENGTH_TELE - FOCAL_LENGTH_WIDE)
            # HFoV/2 in degrees = degrees(atan(HA / (2·FL)))  (≈29.75° at wide end)
            half_hfov = math.degrees(math.atan(HORIZONTAL_APERTURE / (2.0 * fl)))
            # ENU: +yaw = CCW = pan left; +pitch = nose down. Flip the Jetson's
            # frozen.fov_sign_x / fov_sign_y if a lock-on centers the wrong way.
            self._pan  -= float(dp) * half_hfov
            self._tilt -= float(dt) * half_hfov
        self._update_gimbal()

    def _apply_zoom(self, target):
        """Set zoom target; _zoom_loop() will slew smoothly toward it."""
        with self._state_lock:
            self._zoom_target = float(max(0.0, min(1.0, target)))
        print(f"[Host] Zoom target → {self._zoom_target:.3f}")

    def _stop_motion(self):
        """Halt continuous motion. Unlike the old stop_all, this does NOT shut the
        simulator down — the Jetson sends stop_all on every quit, and the
        (slow-to-start) Isaac sim should survive Jetson restarts."""
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        with self._state_lock:
            self._zoom_target = self._zoom   # stop slewing toward an old target
        print("[Host] stop_all → motion halted (sim stays running)")

    def _publish_zoom(self):
        """Publish current _zoom value to the /isaac_core/zoom topic."""
        msg = Float32()
        msg.data = self._zoom
        try:
            self._zoom_pub.publish(msg)
        except Exception as e:
            print(f"[Host] zoom publish error: {e}")

    # ------------------------------------------------------------------
    # 25 Hz ContinuousMove integrator
    # ------------------------------------------------------------------

    def _mover_loop(self):
        """Keeps moving at the coalesced velocity until the command slot goes stale."""
        DT = 0.04  # 25 Hz
        while self._run:
            with self._cmd_lock:
                vx, vy = self._cmd_vel
                age = time.time() - self._cmd_t

            stale = (vx == 0.0 and vy == 0.0) or age > self._cmd_ttl

            if not stale and not self._slewing_home:
                with self._state_lock:
                    self._pan  -= vx * self._pan_vel  * DT   # ENU: +yaw = CCW = pan left
                    self._tilt -= vy * self._tilt_vel * DT   # ENU: +pitch = nose down
                self._update_gimbal()

            time.sleep(DT)

    # ------------------------------------------------------------------
    # 20 Hz zoom slew loop
    # ------------------------------------------------------------------

    def _zoom_loop(self):
        """Moves _zoom toward _zoom_target at 1/7 normalised units per second (matching
        the real camera's zoom_full_s=7.0), publishing whenever the value changes."""
        DT = 0.05           # 20 Hz
        MAX_STEP = DT / 7.0 # ≈ 0.00714 per tick → full range takes 7 s
        while self._run:
            with self._state_lock:
                if self._slewing_home:
                    # slew_home_thread owns _zoom and publishing while slewing
                    delta = 0.0
                else:
                    target = self._zoom_target
                    current = self._zoom
                    delta = target - current
                    if abs(delta) > 1e-6:
                        step = max(-MAX_STEP, min(MAX_STEP, delta))
                        self._zoom = current + step
                    else:
                        delta = 0.0
                        self._zoom = target
            if abs(delta) > 1e-6:
                self._publish_zoom()
            time.sleep(DT)

    # ------------------------------------------------------------------
    # Home logic
    # ------------------------------------------------------------------

    def _save_home(self):
        with self._state_lock:
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
        # Stop any ongoing ContinuousMove before slewing
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        print("[Host] Slewing back to home position...")
        threading.Thread(target=self._slew_home_thread, daemon=True).start()

    def _slew_home_thread(self):
        self._slewing_home = True

        with self._state_lock:
            start_pan = self._pan
            start_tilt = self._tilt
            start_zoom = self._zoom

        end_pan = self._home_pan
        end_tilt = self._home_tilt
        end_zoom = self._home_zoom

        duration = 2.0
        steps = int(duration / 0.02)

        for i in range(steps):
            if not self._run:
                break
            s = 0.5 - 0.5 * math.cos(math.pi * i / steps)  # ease-in-out
            with self._state_lock:
                self._pan = start_pan + (end_pan - start_pan) * s
                self._tilt = start_tilt + (end_tilt - start_tilt) * s
                self._zoom = start_zoom + (end_zoom - start_zoom) * s
            self._update_gimbal()
            self._publish_zoom()
            time.sleep(0.02)

        # Sync target so the zoom loop doesn't re-slew after home completes
        with self._state_lock:
            self._zoom_target = end_zoom
        print("[Host] Home reached.")
        self._slewing_home = False

    # ------------------------------------------------------------------
    # Apply gimbal angles to Isaac Sim
    # ------------------------------------------------------------------

    def _update_gimbal(self):
        """Publish roll/pitch/yaw via persistent publisher — no per-call node creation."""
        msg = Gimbal()
        msg.roll = self._roll
        msg.pitch = self._tilt
        msg.yaw = self._pan
        try:
            self._gimbal_pub.publish(msg)
        except Exception as e:
            print(f"[Host] gimbal publish error: {e}")

    # ------------------------------------------------------------------
    # Move camera to initial position
    # ------------------------------------------------------------------

    def _init_camera_position(self):
        # Send at 30 Hz for 3 s so the packet lands regardless of when Isaac Sim's
        # UDP receiver finishes initialising after the HostIsaacManager ready signal.
        sender = OnePointSender(lat=32.20647, lon=35.29034, alt=540.0,
                                roll=0.0, pitch=0.0, yaw=90.0)
        sender.run(blocking=False)
        time.sleep(3.0)
        sender.stop()
        sender.close()

    # ------------------------------------------------------------------
    # Scene playback
    # ------------------------------------------------------------------

    def _scene_loop(self):
        """Play the Jetson-requested scene once per NEW session.

        Started right after Isaac loads, so the first scene runs ~scene_start_delay_s
        after load. Each new Jetson session (new sim_motion launch → new session
        token) re-arms playback, so quit → change sim.scene → re-run replays it.
        The delay before each play lets the (re)launched pipeline warm up first.
        """
        last_played_session = None
        while self._run:
            session = self._session_id
            if session is None or session == last_played_session:
                time.sleep(0.2)
                continue

            # New session detected — arm playback for it after a settle delay.
            last_played_session = session
            t_arm = time.time()
            while self._run and time.time() - t_arm < self._scene_start_delay_s:
                time.sleep(0.2)
            if not self._run:
                return

            scene = self._requested_scene
            if not scene:   # 0 (or None) → scene playback disabled for this session
                print(f"[Host] session {session}: scene playback disabled (sim.scene=0)")
                continue
            print(f"[Host] starting scene {scene} for session {session} "
                  f"({self._scene_start_delay_s:.0f}s settle)")
            try:
                run_scene(scene)   # blocking; finite trajectory
            except Exception as e:
                print(f"[Host] scene {scene} error: {e}")

    # ------------------------------------------------------------------
    # Pose updates back to Jetson
    # ------------------------------------------------------------------

    def _pose_loop(self):
        while self._run:
            # Explicit override (directly-routable setup) wins; otherwise reply to
            # the source of the last received packet (NAT/VPN case — the default).
            if self.jetson_addr[0] is not None:
                addr = self.jetson_addr
            else:
                with self._addr_lock:
                    addr = self._last_cmd_addr
            if addr is not None:
                packet = {
                    "type": "pose",
                    "pan": self._pan,
                    "tilt": self._tilt,
                    "zoom": self._zoom,
                    "timestamp": time.time(),
                }
                try:
                    # Send from the command socket so the reply matches the NAT
                    # conntrack entry and reaches the Jetson.
                    self.sock.sendto(
                        json.dumps(packet).encode("utf-8"),
                        addr,
                    )
                except Exception:
                    pass
            time.sleep(0.05)  # 20 Hz


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import os

    CORE_PATH = "/home/ofer/clones/ptz_sim"
    USD_PATH = "./usd/maps/earth/earth.usda"

    # Pose is returned by reply-to-sender by default: the Jetson sits behind a
    # NAT/VPN bridge and its 192.168.55.1 is unreachable from here, so we reply to
    # the source address of received commands. Set PTZ_JETSON_IP only for a
    # directly-routable test setup where an explicit pose target is wanted.
    JETSON_IP = os.environ.get("PTZ_JETSON_IP")            # None → reply-to-sender
    JETSON_PORT = int(os.environ.get("PTZ_JETSON_PORT", "5006"))

    # Seconds to wait after Isaac finishes loading before playing the scene the
    # Jetson requested. Override with PTZ_SCENE_DELAY_S.
    SCENE_START_DELAY_S = float(os.environ.get("PTZ_SCENE_DELAY_S", "20.0"))

    sim = PTZSim(
        listen_port=5005,
        jetson_ip=JETSON_IP,
        jetson_port=JETSON_PORT,
        core_path=CORE_PATH,
        usd_path=USD_PATH,
        show_isaac_logs=False,
        image_rtp=True,
        scene_start_delay_s=SCENE_START_DELAY_S,
    )
    sim.run()
