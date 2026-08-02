#!/usr/bin/env python3
import socket
import json
import threading
import time
import math

import rclpy
from isaac_ros2_messages.msg import Gimbal, FrameBboxes
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32

from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.one_point_sender import OnePointSender
import isaac_core_dev_kit.dev_utils as core_utils

# Scene library + the simulated target drone (sibling modules). Run as
# `python3 ./ptz/ptz_sim.py` puts ptz/ on sys.path[0]; fall back to the package
# path if imported differently.
try:
    import scenes
    from drone import SimDrone
except ImportError:
    from ptz import scenes
    from ptz.drone import SimDrone

# Camera model — every physical constant and curve lives in netz250.py.
try:
    import netz250 as cam_model
except ImportError:
    from ptz import netz250 as cam_model

# The bbox publisher reports every child of /bboxes; this is the target drone.
DRONE_TARGET_NAME = "full_drone"
BBOX_TOPIC = "/isaac_core/bbox"


class PTZSim:
    """
    Host-side emulation of the Netz-250 PTZ camera.

    Presents the same control surface the real camera exposes over ONVIF, so
    DroneTracker's PTZSimController can mirror PTZController one-for-one.

    State is held in the camera's own units, NOT in degrees:
      - pan / tilt : ONVIF normalised [-1, +1]  (see netz250.PAN/TILT_DEG_PER_UNIT)
      - zoom       : RAW ONVIF [-1, +1]         (the algorithm normalises to [0,1])

    Faithful behaviours (see netz250.py for the numbers and the reasoning):
      - AbsoluteMove snaps pan/tilt onto a coarse ~0.02 unit grid (real quirk).
      - RelativeMove is near-exact, which is why fine centering uses it.
      - Field of view follows the MEASURED pan-shift curve, not the camera's
        overstated on-screen magnification.
      - Tilt is mechanically limited; both axes clamp to [-1, +1].
      - Zoom slews over ~4 s end-to-end.
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
        bbox_publisher=False,
        cmd_ttl=0.15,   # seconds before a coalesced move command goes stale
        drone_speed_ms=10.0,       # target-drone speed at full stick (manual flight)
    ):
        self.listen_port = listen_port
        # Optional explicit pose target for a directly-routable setup. Default
        # (jetson_ip=None) → reply-to-sender: the sim cannot initiate to the
        # Jetson (it sits behind a NAT/VPN bridge; 192.168.55.1 is unreachable
        # here), so pose is sent back to the source address of received packets.
        self.jetson_addr = (jetson_ip, jetson_port)
        self._last_cmd_addr = None
        self._addr_lock = threading.Lock()

        # PTZ state — protected by _state_lock for R/M/W operations.
        # pan/tilt in ONVIF units; zoom in RAW ONVIF units.
        self._pan = 0.0
        self._tilt = 0.0
        self._roll = 0.0
        self._zoom_raw = cam_model.ZOOM_RAW_MIN          # wide end
        self._zoom_target_raw = cam_model.ZOOM_RAW_MIN   # target the slew chases
        self._state_lock = threading.Lock()

        # Home state (pan/tilt in ONVIF units, zoom RAW)
        self._home_pan = None
        self._home_tilt = None
        self._home_zoom_raw = None

        # ContinuousMove coalescing slot.  Datasheet slew rates, converted from
        # deg/s into ONVIF units/s so the integrator works in camera units.
        self._pan_units_per_s  = cam_model.PAN_SPEED_DEG_S  / cam_model.PAN_DEG_PER_UNIT
        self._tilt_units_per_s = cam_model.TILT_SPEED_DEG_S / cam_model.TILT_DEG_PER_UNIT
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
        # The simulated target drone.  It needs the camera's live pan so that
        # "forward"/"right" stay relative to wherever the camera is looking.
        self._drone = SimDrone(pan_getter=lambda: self._pan,
                              speed_ms=drone_speed_ms)
        # Latest telemetry for the drone, harvested from the bbox publisher.
        self._drone_tlm = None
        self._drone_tlm_lock = threading.Lock()
        self._last_drone_vel = None   # last logged manual-flight direction
        self._bbox_msgs = 0        # bbox frames received from Isaac
        self._drone_tlm_n = 0      # of those, how many carried the target drone

        # Isaac/ROS setup — must happen before threads start so _update_gimbal is safe
        core_utils.delete_cesium_cache()
        core_utils.safe_rclpy_init()
        self._gimbal_node = rclpy.create_node("ptz_sim_gimbal_node")
        self._gimbal_pub = self._gimbal_node.create_publisher(Gimbal, "/isaac_core/gimbal", 10)
        self._zoom_pub   = self._gimbal_node.create_publisher(Float32, "/isaac_core/zoom", 10)
        # Target-drone telemetry.  bbox_node publishes every child of /bboxes with
        # distance_x/y/z already computed as (target - camera) in world metres, so
        # the range readout comes straight off this topic.
        #
        # The QoS must be BEST_EFFORT.  bbox_node publishes with
        # qos_profile_sensor_data (BEST_EFFORT), and a RELIABLE subscriber is
        # INCOMPATIBLE with a BEST_EFFORT publisher — DDS silently delivers
        # nothing, so the topic looks healthy under `ros2 topic echo` (which
        # adapts) while this node receives zero frames.  Measured: RELIABLE got
        # 0 frames in 6 s, BEST_EFFORT got 3.
        bbox_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._bbox_sub = self._gimbal_node.create_subscription(
            FrameBboxes, BBOX_TOPIC, self._on_bboxes, bbox_qos)
        threading.Thread(target=self._ros_spin_loop, daemon=True,
                         name="ptz-sim-ros-spin").start()

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
            bbox_publisher=bbox_publisher,
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
                # Push the starting zoom so Isaac's zoom_node sets the wide-end
                # focal length immediately.  Without this the camera keeps
                # whatever focal length the USD was authored with until the first
                # zoom command arrives, so the initial FOV would be wrong.
                self._publish_zoom()
                # Auto-capture home at the startup pose (0/0/0), mirroring the
                # real controller's connect-time home capture — so the Jetson's
                # automatic go_home (R-key, TRACK→IDLE reset) works from the start.
                self._save_home()
                # The target drone is NOT spawned yet: it appears in front of the
                # camera only when the operator asks for it (scene 1-9, or 0 for
                # manual flight).  Spawning eagerly would put a drone in shot on
                # every boot.
                print(f"[Host] target drone on demand — press 1-"
                      f"{max(scenes.available_scenes())} for a scene, "
                      f"0 for manual numpad flight")
                while self._run:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("[Host] KeyboardInterrupt, shutting down.")
            finally:
                self._run = False
                try:
                    self._drone.close()
                except Exception:
                    pass
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
            self._handle_cmd(msg, addr)

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    def _handle_cmd(self, msg, addr):
        t = msg.get("type")
        if t == "ping":
            return   # heartbeat: only refreshes the reply address (done in _recv_loop)
        if t == "drone_vel":
            # Can arrive on every key repeat, so log only when the direction
            # actually changes — enough to prove commands are arriving without
            # flooding the console.
            vec = (msg.get("fwd", 0.0), msg.get("right", 0.0), msg.get("up", 0.0))
            if vec != self._last_drone_vel:
                self._last_drone_vel = vec
                print(f"[Host] drone_vel fwd={vec[0]:+.0f} right={vec[1]:+.0f} "
                      f"up={vec[2]:+.0f}")
        else:
            print(f"[Host] From {addr}: {msg}")

        if t == "move":
            # ONVIF ContinuousMove — normalised velocity, TTL-coalesced.
            self._apply_move(msg["vx"], msg["vy"])
        elif t == "relative_move":
            # ONVIF RelativeMove — near-exact small delta, no quantisation.
            self._apply_relative(msg["dp"], msg["dt"])
        elif t == "center_fine":
            # Fine centering: RelativeMove, deliberately NOT quantised. This is
            # what makes PTZController.center_fine() accurate at high zoom.
            self._apply_relative(msg["dp"], msg["dt"])
        elif t == "zoom_to":
            # RAW ONVIF zoom target (the algorithm denormalises before sending).
            self._apply_zoom_raw(msg["raw"] if "raw" in msg else msg["target"])
        elif t == "absolute_move":
            # ONVIF AbsoluteMove — pan/tilt snap to the coarse grid; any of the
            # three axes may be omitted (None) to leave it untouched.
            self._apply_absolute(msg.get("pan"), msg.get("tilt"), msg.get("zoom"))
        elif t == "center_and_zoom":
            if msg.get("use_absolute"):
                # Netz-250 path: read-modify-write into ONE AbsoluteMove, exactly
                # as the real controller does (pan_now + dp, tilt_now + dt, zoom).
                with self._state_lock:
                    pan_now, tilt_now = self._pan, self._tilt
                self._apply_absolute(pan_now + float(msg["dp"]),
                                     tilt_now + float(msg["dt"]),
                                     msg.get("zoom_raw"))
            else:
                # Legacy IPC6852 path: RelativeMove + separate zoom.
                self._apply_relative(msg["dp"], msg["dt"])
                if msg.get("zoom_raw") is not None:
                    self._apply_zoom_raw(msg["zoom_raw"])
        elif t == "save_home":
            self._save_home()
        elif t == "go_home":
            self._go_home(msg.get("zoom_raw"))
        elif t == "stop_all":
            self._stop_motion()

        # ── Target-drone control (simulator-only; no real-camera equivalent) ──
        elif t == "scene":
            # 0 = manual joystick mode, 1..9 = play that canned scene.
            num = int(msg.get("n", 0))
            if num == 0:
                self._drone.set_joystick_mode(True)
            else:
                self._drone.play_scene(num)
        elif t == "drone_vel":
            # Camera-relative stick, each component -1..+1.
            self._drone.set_velocity(msg.get("fwd", 0.0),
                                     msg.get("right", 0.0),
                                     msg.get("up", 0.0))
        elif t == "drone_stop":
            self._drone.stop_scene()
            self._drone.set_joystick_mode(False)

    # ------------------------------------------------------------------
    # PTZ motion logic
    # ------------------------------------------------------------------

    def _apply_move(self, vx, vy):
        """ONVIF ContinuousMove: write normalised velocity to the coalescing slot.

        ``vx``/``vy`` are ONVIF velocities in [-1,+1]; the integrator converts
        them to units/s using the datasheet slew rates.
        """
        with self._cmd_lock:
            self._cmd_vel = (float(vx), float(vy))
            self._cmd_t = time.time()

    def _apply_relative(self, dp, dt):
        """ONVIF RelativeMove: add an exact delta in ONVIF pan/tilt units.

        The real camera executes these near-exactly (0.005 commanded → 0.0049
        measured), which is why DroneTracker uses RelativeMove for fine
        centering.  Deliberately NOT snapped to the AbsoluteMove grid.  Clears
        the coalescing slot so the mover doesn't fight it.
        """
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        acc = cam_model.RELATIVE_MOVE_ACCURACY
        with self._state_lock:
            self._pan = cam_model.clamp_pan(self._pan + float(dp) * acc)
            self._tilt = cam_model.clamp_tilt(self._tilt + float(dt) * acc)
        self._update_gimbal()

    def _apply_absolute(self, pan=None, tilt=None, zoom_raw=None):
        """ONVIF AbsoluteMove: jump to an absolute pose.

        Reproduces the real camera's quirk — pan/tilt targets snap onto a coarse
        ~0.02 unit grid, so deltas smaller than half a grid step vanish entirely.
        Zoom is exact and slews via _zoom_loop.
        """
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        with self._state_lock:
            if pan is not None:
                self._pan = cam_model.clamp_pan(
                    cam_model.quantize_absolute(float(pan)))
            if tilt is not None:
                self._tilt = cam_model.clamp_tilt(
                    cam_model.quantize_absolute(float(tilt)))
            if zoom_raw is not None:
                self._zoom_target_raw = max(cam_model.ZOOM_RAW_MIN,
                                            min(cam_model.ZOOM_RAW_MAX, float(zoom_raw)))
            pan_q, tilt_q, ztgt = self._pan, self._tilt, self._zoom_target_raw
        print(f"[Host] AbsoluteMove → pan={pan_q:+.4f} tilt={tilt_q:+.4f} "
              f"zoom_raw={ztgt:+.3f} (pan/tilt snapped to {cam_model.ABSOLUTE_MOVE_GRID} grid)")
        self._update_gimbal()

    def _apply_zoom_raw(self, raw_target):
        """Set the RAW zoom target; _zoom_loop slews toward it."""
        with self._state_lock:
            self._zoom_target_raw = max(cam_model.ZOOM_RAW_MIN,
                                        min(cam_model.ZOOM_RAW_MAX, float(raw_target)))
            tgt = self._zoom_target_raw
        print(f"[Host] Zoom target → raw {tgt:+.3f} "
              f"(norm {cam_model.normalize_zoom(tgt):.3f}, "
              f"{cam_model.true_magnification(cam_model.normalize_zoom(tgt)):.1f}x)")

    def _stop_motion(self):
        """Halt continuous motion. Unlike the old stop_all, this does NOT shut the
        simulator down — the Jetson sends stop_all on every quit, and the
        (slow-to-start) Isaac sim should survive Jetson restarts."""
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t = 0.0
        with self._state_lock:
            self._zoom_target_raw = self._zoom_raw   # stop slewing toward an old target
        print("[Host] stop_all → motion halted (sim stays running)")

    def _publish_zoom(self):
        """Publish zoom to Isaac on /isaac_core/zoom as NORMALISED [0,1].

        zoom_node.py maps that to a focal length using the same measured curve.
        """
        msg = Float32()
        with self._state_lock:
            msg.data = float(cam_model.normalize_zoom(self._zoom_raw))
        try:
            self._zoom_pub.publish(msg)
        except Exception as e:
            print(f"[Host] zoom publish error: {e}")

    # ------------------------------------------------------------------
    # 25 Hz ContinuousMove integrator
    # ------------------------------------------------------------------

    def _mover_loop(self):
        """Integrates the coalesced ContinuousMove velocity until it goes stale.

        Works in ONVIF units/s (datasheet deg/s ÷ deg-per-unit).  The sign
        convention matches the real camera's ``continuous_pan_sign = -1`` /
        ``continuous_tilt_sign = -1``: a POSITIVE vx makes the reported pan
        DECREASE.  DroneTracker's config carries those same signs, so the two
        agree; flip them there if a manual pan goes the wrong way.
        """
        DT = 0.04  # 25 Hz
        while self._run:
            with self._cmd_lock:
                vx, vy = self._cmd_vel
                age = time.time() - self._cmd_t

            stale = (vx == 0.0 and vy == 0.0) or age > self._cmd_ttl

            if not stale and not self._slewing_home:
                with self._state_lock:
                    self._pan = cam_model.clamp_pan(
                        self._pan - vx * self._pan_units_per_s * DT)
                    self._tilt = cam_model.clamp_tilt(
                        self._tilt - vy * self._tilt_units_per_s * DT)
                self._update_gimbal()

            time.sleep(DT)

    # ------------------------------------------------------------------
    # Zoom slew loop
    # ------------------------------------------------------------------

    def _zoom_loop(self):
        """Slews RAW zoom toward the target at the camera's real travel rate.

        ZOOM_FULL_TRAVEL_S covers the FULL raw span, matching the datasheet /
        DroneTracker's ``zoom.full_travel_s`` so lock-on settle timing lines up.
        """
        DT = 0.05                                             # 20 Hz
        span = cam_model.ZOOM_RAW_MAX - cam_model.ZOOM_RAW_MIN
        MAX_STEP = span * DT / cam_model.ZOOM_FULL_TRAVEL_S
        while self._run:
            with self._state_lock:
                if self._slewing_home:
                    # slew_home_thread owns zoom and publishing while slewing
                    delta = 0.0
                else:
                    delta = self._zoom_target_raw - self._zoom_raw
                    if abs(delta) > 1e-6:
                        step = max(-MAX_STEP, min(MAX_STEP, delta))
                        self._zoom_raw += step
                    else:
                        delta = 0.0
                        self._zoom_raw = self._zoom_target_raw
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
            self._home_zoom_raw = self._zoom_raw
        print(f"[Host] Home saved: pan={self._home_pan:+.4f}, tilt={self._home_tilt:+.4f}, "
              f"zoom_raw={self._home_zoom_raw:+.3f}")

    def _go_home(self, zoom_raw=None):
        """Slew back to the saved home pose.

        ``zoom_raw`` lets the Jetson specify the zoom to restore, mirroring
        PTZController._do_go_home's choice of (home_zoom → pre-lock zoom →
        search position).  Falls back to the locally saved home zoom.
        """
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
        threading.Thread(target=self._slew_home_thread, args=(zoom_raw,),
                         daemon=True).start()

    def _slew_home_thread(self, zoom_raw=None):
        self._slewing_home = True

        with self._state_lock:
            start_pan = self._pan
            start_tilt = self._tilt
            start_zoom = self._zoom_raw

        end_pan = self._home_pan
        end_tilt = self._home_tilt
        end_zoom = (float(zoom_raw) if zoom_raw is not None
                    else self._home_zoom_raw)

        duration = 2.0
        steps = int(duration / 0.02)

        for i in range(steps):
            if not self._run:
                break
            s = 0.5 - 0.5 * math.cos(math.pi * i / steps)  # ease-in-out
            with self._state_lock:
                self._pan = cam_model.clamp_pan(start_pan + (end_pan - start_pan) * s)
                self._tilt = cam_model.clamp_tilt(start_tilt + (end_tilt - start_tilt) * s)
                self._zoom_raw = start_zoom + (end_zoom - start_zoom) * s
            self._update_gimbal()
            self._publish_zoom()
            time.sleep(0.02)

        # Sync target so the zoom loop doesn't re-slew after home completes
        with self._state_lock:
            self._zoom_target_raw = end_zoom
        print("[Host] Home reached.")
        self._slewing_home = False

    # ------------------------------------------------------------------
    # Apply gimbal angles to Isaac Sim
    # ------------------------------------------------------------------

    def _update_gimbal(self):
        """Publish roll/pitch/yaw to Isaac, converting ONVIF units → degrees.

        Sign convention — derived from the algorithm's own tuning, do not "fix"
        one half of it in isolation:

          * increasing ONVIF pan  = camera looks RIGHT
          * increasing ONVIF tilt = camera looks UP

        Why: lock-on centres a target at normalised offset ex>0 (right of centre)
        with ``dx = fov_sign_x · gain · ex / spp`` and ``fov_sign_x = +1``, i.e. a
        POSITIVE pan delta must move the view toward a target on the right.
        Likewise ``fov_sign_y = -1`` means a target below centre (ey>0) yields a
        NEGATIVE tilt delta, so decreasing tilt must look down.

        Isaac's ENU gimbal is the opposite handedness on both axes (+yaw = CCW =
        left, +pitch = nose down), hence the negation here.  Pan/tilt stay LINEAR
        in ONVIF units — the zoom dependence lives in the field of view, not in
        the pan gain (see netz250.py).
        """
        msg = Gimbal()
        with self._state_lock:
            msg.roll = self._roll
            msg.pitch = -self._tilt * cam_model.TILT_DEG_PER_UNIT
            msg.yaw = -self._pan * cam_model.PAN_DEG_PER_UNIT
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
        cam = scenes.CAMERA
        sender = OnePointSender(lat=cam["lat"], lon=cam["lon"], alt=cam["alt"],
                                roll=0.0, pitch=0.0, yaw=cam["yaw_deg"])
        sender.run(blocking=False)
        time.sleep(3.0)
        sender.stop()
        sender.close()

    # ------------------------------------------------------------------
    # Scene playback
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Target-drone telemetry (from the bbox publisher)
    # ------------------------------------------------------------------

    def _ros_spin_loop(self):
        """Spin the ROS node so the bbox subscription delivers callbacks.

        Also reports the state of the telemetry stream every 10 s, because "no
        range readout" has three very different causes and they are otherwise
        indistinguishable from the GUI.
        """
        last_report = time.time()
        last_msgs = 0
        while self._run:
            try:
                rclpy.spin_once(self._gimbal_node, timeout_sec=0.1)
            except Exception:
                time.sleep(0.1)

            now = time.time()
            if now - last_report < 10.0:
                continue
            last_report = now
            if self._bbox_msgs == last_msgs:
                if self._bbox_msgs == 0:
                    print(f"[Host] no bbox messages on {BBOX_TOPIC} — the range "
                          f"readout will stay empty. Is --bbox-publisher enabled "
                          f"and is the bbox script node running?")
                else:
                    print(f"[Host] bbox stream stalled at {self._bbox_msgs} frames")
            elif self._drone_tlm_n == 0:
                print(f"[Host] {self._bbox_msgs} bbox frames received but none "
                      f"contained '{DRONE_TARGET_NAME}' — no range readout")
            last_msgs = self._bbox_msgs

    def _on_bboxes(self, msg):
        """Pick the target drone out of the frame's bboxes and cache its offset.

        ``distance_x/y/z`` come from bbox_node as (target_pos - camera_pos) in
        Isaac world metres (ENU: x=east, y=north, z=up).  This is CLOSED-LOOP
        data: it is the drone's actual rendered world pose, not where we told it
        to go, so it reflects whatever Isaac really did.

        NOTE: the message's lat/lon/alt are read from the prim's cesium anchor
        attributes, which are NOT updated as the drone flies, so they are ignored
        here — the distances come from live world poses and are trustworthy.
        """
        self._bbox_msgs += 1
        names = [b.target_name for b in msg.bboxes]
        if self._bbox_msgs == 1:
            print(f"[Host] bbox stream up — targets seen: {names or '(none)'}")
            if DRONE_TARGET_NAME not in names:
                print(f"[Host] WARNING: '{DRONE_TARGET_NAME}' is not among the bbox "
                      f"targets, so there will be no range readout. bbox_node only "
                      f"reports children of /bboxes.")

        for b in msg.bboxes:
            if b.target_name != DRONE_TARGET_NAME:
                continue
            east, north, up = float(b.distance_x), float(b.distance_y), float(b.distance_z)
            # Rotate the world offset into the camera's own frame so the operator
            # sees forward/right/up rather than compass directions.
            bearing = math.radians(self._drone_bearing_offset())
            fwd = east * math.sin(bearing) + north * math.cos(bearing)
            right = east * math.cos(bearing) - north * math.sin(bearing)
            with self._drone_tlm_lock:
                self._drone_tlm = {
                    "fwd": fwd, "right": right, "up": up,
                    "range": math.sqrt(east * east + north * north + up * up),
                }
            # NOTE: bbox_node also reports in_frame / is_visible, but Isaac's
            # 2D-bbox visibility is unreliable for this target — it reports "not
            # in frame" even when the drone is plainly centred in the video. The
            # flags are therefore deliberately NOT forwarded; a readout that is
            # always wrong is worse than none. The distances above come from
            # world poses and are trustworthy.
            self._drone_tlm_n += 1
            return

    def _drone_bearing_offset(self):
        """Camera bearing in degrees — same convention SimDrone flies with."""
        from drone import (CAMERA_AZIMUTH_AT_PAN_ZERO_DEG,
                           CAMERA_AZIMUTH_PER_PAN_DEG)
        with self._state_lock:
            pan = self._pan
        return CAMERA_AZIMUTH_AT_PAN_ZERO_DEG + CAMERA_AZIMUTH_PER_PAN_DEG * pan

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
                with self._state_lock:
                    pan, tilt, zoom_raw = self._pan, self._tilt, self._zoom_raw
                packet = {
                    "type": "pose",
                    # ONVIF units, exactly what GetStatus would report.
                    "pan": pan,
                    "tilt": tilt,
                    # RAW ONVIF zoom — PTZController.get_pose() returns raw and
                    # normalises separately, so the sim must match.
                    "zoom": zoom_raw,
                    "timestamp": time.time(),
                }
                # Simulator-only extra: target-drone offset for the GUI readout.
                # Rides along on the pose packet so it needs no extra socket and
                # inherits the same reply-to-sender NAT behaviour.
                with self._drone_tlm_lock:
                    tlm = self._drone_tlm
                packet["drone"] = tlm            # None when no drone/bbox yet
                packet["scene"] = self._drone.scene
                packet["joystick"] = self._drone.joystick_on
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

    # Target-drone speed at full stick during manual (numpad) flight, m/s.
    DRONE_SPEED_MS = float(os.environ.get("PTZ_DRONE_SPEED_MS", "10.0"))

    sim = PTZSim(
        listen_port=5005,
        jetson_ip=JETSON_IP,
        jetson_port=JETSON_PORT,
        core_path=CORE_PATH,
        usd_path=USD_PATH,
        show_isaac_logs=False,
        image_rtp=True,
        bbox_publisher=True,
        drone_speed_ms=DRONE_SPEED_MS,
    )
    sim.run()
