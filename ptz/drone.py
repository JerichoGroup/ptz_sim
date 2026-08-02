"""SimDrone — the simulated target drone inside Isaac Sim.

One long-lived drone that can either fly a canned scene or be flown manually
from the DroneTracker window's flight keys (i/k, j/l, u/o).  Wraps ``UdpBot`` from the dev kit, which
streams pose packets to Isaac on UDP 33335 (driving ``/bboxes/full_drone``).

Everything is expressed **relative to the camera**, because that is the only
frame a human operator can reason about while watching the video:

    forward  = away from the camera        backward = toward the camera
    right    = to the camera's right       left     = to the camera's left
    up/down  = altitude (never affected by camera tilt)

The camera only rotates (it never translates), so "away from the camera" is the
direction the camera is currently panned toward.  ``SimDrone`` asks the host for
that pan angle through the ``pan_getter`` callback, converts the operator's
camera-relative request into world east/north, and moves the drone there.

Manual flight is velocity-based with a timeout, mirroring how the PTZ arrow keys
work: each key press refreshes a velocity that expires shortly after the key is
released, so holding a key flies smoothly and letting go stops.
"""

import math
import threading
import time

from isaac_core_dev_kit.udp.udp_bot import UdpBot
from isaac_core_dev_kit.udp.udp_utils import meters_to_latlon_offset

try:
    import scenes
except ImportError:                                   # pragma: no cover
    from ptz import scenes


# Manual-flight defaults
DEFAULT_SPEED_MS = 10.0     # metres per second while a direction key is held
JOYSTICK_HZ = 30.0          # integration/publish rate while flying manually

# How long one key press keeps the drone moving.
#
# Holding a key only produces smooth flight if the OS key-repeat actually reaches
# cv2.waitKey, which is not guaranteed.  Without repeats a press would move the
# drone for VELOCITY_TTL_S only — at 10 m/s that is 3.5 m, which at a few hundred
# metres of range is visually indistinguishable from nothing happening (this is
# exactly how manual flight first appeared "broken").
#
# So a single press is guaranteed to travel at least MIN_PRESS_TRAVEL_M.  If key
# repeats DO arrive they simply refresh the window and flight stays continuous at
# DEFAULT_SPEED_MS, which is the intended feel.
VELOCITY_TTL_S = 0.35       # minimum hold after the last press
MIN_PRESS_TRAVEL_M = 15.0   # guaranteed travel per press, metres

# ── Where the camera is looking, as a compass bearing ────────────────────────
# "forward" means "away from the camera", so we need the camera's azimuth.
#
#   bearing_deg = AZIMUTH_AT_PAN_ZERO + AZIMUTH_PER_PAN * pan
#
# Derivation for the current scene: _init_camera_position sends the camera pose
# with NED yaw = 90 deg; OgnSimUDPToGlobalPosition converts NED->ENU as
# yaw_enu = -yaw_ned + 90, giving ENU yaw 0 (pointing east).  The gimbal then
# adds -pan*180 deg of ENU yaw, and a compass bearing is 90 - yaw_enu, so:
#   bearing = 90 - (-pan*180) = 90 + pan*180
#
# ►► IF MANUAL FLIGHT GOES THE WRONG WAY, FIX IT HERE — nothing else. ◄◄
#    forward/back swapped with left/right  -> add or subtract 90 from the offset
#    forward and backward inverted         -> add 180 to the offset
#    left and right inverted               -> negate AZIMUTH_PER_PAN
# These two numbers also steer the canned scenes, so both stay consistent.
CAMERA_AZIMUTH_AT_PAN_ZERO_DEG = 90.0
CAMERA_AZIMUTH_PER_PAN_DEG = 180.0


class SimDrone:
    """The simulated target drone: canned scenes + manual keyboard flight."""

    def __init__(self, pan_getter=None, speed_ms=DEFAULT_SPEED_MS,
                 udp_port=scenes.DRONE_UDP_PORT, send_rate_hz=30.0):
        """
        Args:
            pan_getter: callable returning the camera's current pan in ONVIF
                units [-1,+1].  Used to rotate camera-relative requests into
                world east/north.  Defaults to "camera pointing at pan 0".
            speed_ms:   metres per second at full stick during manual flight.
        """
        self._pan_getter = pan_getter or (lambda: 0.0)
        self._speed_ms = float(speed_ms)
        self._udp_port = udp_port
        self._send_rate_hz = send_rate_hz

        self._bot = None
        self._lock = threading.Lock()

        # Manual-flight state
        self._vel = (0.0, 0.0, 0.0)      # (forward, right, up), each -1..+1
        self._vel_until = 0.0            # keep moving until this monotonic time
        self._joystick_on = False
        self._flown_m = 0.0              # metres travelled since the last log
        self._last_fly_log = 0.0

        # Scene playback state
        self._scene_num = None           # scene currently playing, or None
        self._scene_gen = 0              # bumped to cancel a running scene
        self._scene_thread = None

        self._run = True
        self._joy_thread = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Create the drone and begin streaming its pose to Isaac.

        Called lazily on the first scene / manual-flight command — NOT at
        startup.  The drone spawns at the camera's look-point, so spawning it
        eagerly parks it right on top of the lens and the video goes black.
        Until the operator asks for a target, there simply isn't one.
        """
        if self._bot is not None:
            return
        lat, lon, alt = self._spawn_lla()
        self._bot = UdpBot(
            udp_port=self._udp_port,
            send_rate_hz=self._send_rate_hz,
            start_lat=lat,
            start_lon=lon,
            start_alt=alt,
            start_roll_d=0.0,
            start_pitch_d=0.0,
            start_yaw_d=0.0,
        )
        self._bot.run(blocking=False)
        if self._joy_thread is None:
            self._joy_thread = threading.Thread(
                target=self._joystick_loop, daemon=True, name="sim-drone-joystick")
            self._joy_thread.start()
        off = scenes.SPAWN_OFFSET
        print(f"[Drone] spawned {off['forward']:.0f} m in front of the camera "
              f"({lat:.5f},{lon:.5f} alt {alt:.0f} m, UDP {self._udp_port})")

    @property
    def spawned(self):
        """True once the drone exists (i.e. a scene or manual flight was asked for)."""
        return self._bot is not None

    def close(self):
        """Stop everything and release the socket."""
        self._run = False
        self.stop_scene()
        if self._bot is not None:
            try:
                self._bot.stop()
                self._bot.close()
            except Exception:
                pass
            self._bot = None

    # ── camera-relative geometry ─────────────────────────────────────────────

    def _camera_bearing_rad(self):
        """Camera heading as a compass bearing (0 = north, clockwise), in radians.

        Derived from the camera's ONVIF pan — see the constants at the top of this
        module, which are the single place to fix a wrong direction.  Tilt is
        deliberately ignored: altitude is the operator's job via up/down.
        """
        deg = (CAMERA_AZIMUTH_AT_PAN_ZERO_DEG
               + CAMERA_AZIMUTH_PER_PAN_DEG * float(self._pan_getter()))
        return math.radians(deg)

    def _to_east_north(self, forward_m, right_m):
        """Rotate a camera-relative (forward, right) offset into (east, north)."""
        b = self._camera_bearing_rad()
        # bearing is clockwise from north: north = cos(b), east = sin(b);
        # "right" is 90 deg clockwise of forward.
        east = forward_m * math.sin(b) + right_m * math.cos(b)
        north = forward_m * math.cos(b) - right_m * math.sin(b)
        return east, north

    def _spawn_lla(self):
        """Where the drone appears: SPAWN_OFFSET applied from the camera.

        Camera-relative, so the drone is always in view no matter where the
        camera happens to be panned when the operator asks for a target.
        """
        cam = scenes.CAMERA
        off = scenes.SPAWN_OFFSET
        east, north = self._to_east_north(off["forward"], off["right"])
        d_lat, d_lon = meters_to_latlon_offset(north, east, cam["lat"])
        return (cam["lat"] + d_lat, cam["lon"] + d_lon, cam["alt"] + off["up"])

    # ── manual flight (i/k, j/l, u/o) ───────────────────────────────────────

    def set_velocity(self, forward, right, up):
        """Set the manual-flight velocity, each component in -1..+1.

        Refreshed on every key press; expires after VELOCITY_TTL_S so the drone
        stops shortly after the key is released.  Ignored unless joystick mode is
        on (so a stray key can't disturb a scene).
        """
        with self._lock:
            if not self._joystick_on:
                print("[Drone] ignoring drone_vel — manual flight is OFF "
                      "(press 0 in the DroneTracker window first)")
                return
            moving = not (forward == 0.0 and right == 0.0 and up == 0.0)
            hold = max(VELOCITY_TTL_S, MIN_PRESS_TRAVEL_M / max(self._speed_ms, 1e-6))
            self._vel = (float(forward), float(right), float(up))
            self._vel_until = time.time() + (hold if moving else 0.0)

    def set_joystick_mode(self, enabled):
        """Turn manual flight on/off.  Turning it on cancels any running scene."""
        enabled = bool(enabled)
        if enabled:
            self.start()          # spawn on demand — see start()'s docstring
            self.stop_scene()
        with self._lock:
            self._joystick_on = enabled
            self._vel = (0.0, 0.0, 0.0)
            self._vel_t = 0.0
        print(f"[Drone] joystick mode {'ON' if enabled else 'off'}")

    @property
    def joystick_on(self):
        with self._lock:
            return self._joystick_on

    def _joystick_loop(self):
        """Integrate the manual velocity into the drone's position."""
        dt = 1.0 / JOYSTICK_HZ
        while self._run:
            time.sleep(dt)
            now = time.time()
            with self._lock:
                if not self._joystick_on or self._bot is None:
                    continue
                fwd, right, up = self._vel
                stale = now > self._vel_until
            if stale or (fwd == 0.0 and right == 0.0 and up == 0.0):
                continue

            step = self._speed_ms * dt
            east, north = self._to_east_north(fwd * step, right * step)
            self._nudge(north_m=north, east_m=east, up_m=up * step)

            # Report actual travel — the only way to tell "flying" from
            # "commands arriving but nothing moving".
            self._flown_m += step
            if now - self._last_fly_log > 1.0:
                self._last_fly_log = now
                b = self._bot
                print(f"[Drone] flying fwd={fwd:+.0f} right={right:+.0f} up={up:+.0f}"
                      f"  travelled {self._flown_m:.1f} m"
                      f"  now {b._current_lat:.5f},{b._current_lon:.5f}"
                      f" alt {b._current_alt:.0f} m")
                self._flown_m = 0.0

    def _nudge(self, north_m, east_m, up_m):
        """Move the drone by a small offset, in metres, and publish immediately."""
        bot = self._bot
        if bot is None:
            return
        try:
            d_lat, d_lon = meters_to_latlon_offset(north_m, east_m, bot._current_lat)
            bot._current_lat += d_lat
            bot._current_lon += d_lon
            bot._current_alt += up_m
            bot._publish_current_pose()
        except Exception as e:
            print(f"[Drone] nudge error: {e}")

    # ── scene playback ───────────────────────────────────────────────────────

    @property
    def scene(self):
        """Scene number currently playing, or None."""
        with self._lock:
            return self._scene_num

    def stop_scene(self):
        """Cancel a running scene (the drone hovers where it is)."""
        with self._lock:
            self._scene_gen += 1
            self._scene_num = None

    def play_scene(self, num, blocking=False):
        """Fly scene ``num``.  Cancels whatever was running first.

        Returns False if the scene number isn't defined.
        """
        if num not in scenes.SCENES:
            print(f"[Drone] scene {num} not defined; available: "
                  f"{scenes.available_scenes()}")
            return False

        self.start()                           # spawn on demand
        self.stop_scene()                      # supersede any running scene
        with self._lock:
            self._joystick_on = False          # a scene owns the drone
            self._scene_gen += 1
            gen = self._scene_gen
            self._scene_num = num

        if blocking:
            self._scene_worker(num, gen)
            return True

        self._scene_thread = threading.Thread(
            target=self._scene_worker, args=(num, gen), daemon=True,
            name=f"sim-drone-scene-{num}")
        self._scene_thread.start()
        return True

    def _superseded(self, gen):
        with self._lock:
            return self._scene_gen != gen or not self._run

    def _scene_worker(self, num, gen):
        """Run a scene's steps in order, bailing out if superseded."""
        scene = scenes.SCENES[num]
        print(f"[Drone] scene {num} start — {scene.get('name', '')}")
        try:
            self._reset_to_spawn()
            for i, step in enumerate(scene["steps"], 1):
                if self._superseded(gen):
                    print(f"[Drone] scene {num} cancelled at step {i}")
                    return
                self._run_step(step, gen)
        except Exception as e:
            print(f"[Drone] scene {num} error: {e}")
        finally:
            with self._lock:
                if self._scene_gen == gen:
                    self._scene_num = None
            print(f"[Drone] scene {num} finished")

    def _reset_to_spawn(self):
        """Put the drone back at the look-point so scenes are repeatable."""
        bot = self._bot
        if bot is None:
            return
        lat, lon, alt = self._spawn_lla()
        bot._current_lat = lat
        bot._current_lon = lon
        bot._current_alt = alt
        bot._publish_current_pose()

    def _run_step(self, step, gen):
        """Execute one scene step (blocking for its duration)."""
        move = step.get("move")
        seconds = float(step.get("seconds", 1.0))

        if move == "wait":
            self._sleep_cancellable(seconds, gen)
            return

        bot = self._bot
        if bot is None:
            return

        if move == "goto":
            self._glide_to(float(step["lat"]), float(step["lon"]),
                           float(step["alt"]), seconds, gen)
            return

        axes = scenes.MOVE_AXES.get(move)
        if axes is None:
            print(f"[Drone] unknown move {move!r} — skipped")
            return

        meters = float(step.get("meters", 0.0))
        f, r, u = axes
        east, north = self._to_east_north(f * meters, r * meters)
        d_lat, d_lon = meters_to_latlon_offset(north, east, bot._current_lat)
        self._glide_to(bot._current_lat + d_lat,
                       bot._current_lon + d_lon,
                       bot._current_alt + u * meters,
                       seconds, gen)

    def _glide_to(self, lat, lon, alt, seconds, gen):
        """Interpolate smoothly to a target pose, checking for cancellation."""
        bot = self._bot
        if bot is None:
            return
        steps = max(1, int(JOYSTICK_HZ * seconds))
        dt = seconds / steps
        lat0, lon0, alt0 = bot._current_lat, bot._current_lon, bot._current_alt
        for i in range(1, steps + 1):
            if self._superseded(gen):
                return
            a = i / steps
            bot._current_lat = lat0 + (lat - lat0) * a
            bot._current_lon = lon0 + (lon - lon0) * a
            bot._current_alt = alt0 + (alt - alt0) * a
            bot._publish_current_pose()
            time.sleep(dt)

    def _sleep_cancellable(self, seconds, gen):
        end = time.time() + seconds
        while time.time() < end:
            if self._superseded(gen):
                return
            time.sleep(0.05)
