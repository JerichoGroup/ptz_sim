#!/usr/bin/env python3
"""Generate a labelled drone-on-sky dataset from the running simulator.

    Terminal 1:  ISAACSIM_PYTHON ./scripts/data_sim.py     (leave running)
    Terminal 2:  ISAACSIM_PYTHON ./scripts/gen_dataset.py

Produces, per sample, in OUT_DIR:

    000123.png    the rendered frame (drone against sky, optionally a bird)
    000123.json   labels + camera state, with field docs embedded

Needs ISAACSIM_PYTHON (not the system python): UdpBot imports transforms3d, which
is broken against the numpy in the user site-packages here.

--------------------------------------------------------------------------------
HOW IT WORKS, AND THE TWO RULES THAT KEEP IT FROM FIGHTING THE SIM
--------------------------------------------------------------------------------
This is a pure CLIENT of the running simulator. It never launches Isaac.

  1. THE CAMERA IS COMMANDED THROUGH ptz_sim, over UDP 5005, exactly the way the
     Jetson does it. Do NOT publish to /isaac_core/gimbal or /isaac_core/zoom
     from here: ptz_sim publishes those continuously from its own loops, and the
     two would fight, leaving the camera twitching between two poses.

  2. THE DRONE IS DRIVEN DIRECTLY, over UDP 33335, with our own UdpBot. Do NOT
     send `scene` / `drone_vel` / `drone_stop` to ptz_sim: those spawn its own
     SimDrone, whose sender loop would then compete with ours for the same prim.
     As long as we never send them, its drone stays unspawned and silent.

WHERE EACH LABEL COMES FROM
    pixel position   /isaac_core/bbox, the "Cube" target. Cube is an 8 cm marker
                     mesh that is a SIBLING of full_drone under /bboxes — and the
                     OmniGraph writes translate/orient to /bboxes itself, i.e. the
                     PARENT, so the whole subtree moves as one and Cube rides
                     with the drone. Its centre offset from the drone origin is
                     ~4 cm, which is well under a pixel at these ranges.
                     (We use Cube rather than full_drone because full_drone's
                     semantic tags sit on its child meshes, so bbox_node's
                     `prims_in_loose.get(path)` lookup misses and it reports -1.)
    range            same message, sqrt(distance_x^2 + y^2 + z^2). Those are
                     computed from live world poses, so they are trustworthy.
    zoom             we command it and confirm it from ptz_sim's pose replies;
                     focal length comes from netz250, the same model the sim's
                     zoom node uses to set the camera.

DELIBERATELY NOT USED: the bbox message's lat/lon/alt. Those read cesium:anchor:*
on the prim, which are NOT updated as the drone moves — they would be silently
wrong labels.

AIM ANALYTICALLY, LABEL FROM ISAAC. The projection maths below only exists to
put the drone near a chosen pixel so the hit rate is high. Every value written to
the JSON is read back from the simulator.
"""

import json
import math
import os
import random
import socket
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "ptz"))

import netz250 as cam_model                                     # noqa: E402
import scenes                                                   # noqa: E402

import rclpy                                                    # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy   # noqa: E402
from isaac_ros2_messages.msg import FrameBboxes, SATOutput       # noqa: E402

from isaac_core_dev_kit.udp.udp_bot import UdpBot                # noqa: E402
from isaac_core_dev_kit.udp.udp_utils import meters_to_latlon_offset  # noqa: E402


# ==============================================================================
# CONFIGURATION — everything you are likely to change lives here
# ==============================================================================

NUM_IMAGES = 200                # how many samples to produce

OUT_DIR = os.path.join(_ROOT, "dataset")
IMAGE_W, IMAGE_H = 1920, 1080   # must match the sim's render resolution

# ---- camera pose ------------------------------------------------------------
ZOOM_NORM = 0.30                # fixed for now (HFoV 40.0 deg, focal 7.65 mm)

# Tilt UP so the whole frame is sky. At ZOOM_NORM=0.30 the vertical FoV is
# 23.1 deg, so 0.56 units (25.2 deg) leaves the bottom edge ~13.6 deg above the
# horizon — a comfortable margin, no terrain.
TILT_UNITS = 0.56

# Pan is randomised per sample purely for background variety; with the camera
# this far up, every direction is sky.
PAN_RANGE = (-1.0, 1.0)

# ---- where the drone goes ---------------------------------------------------
# The drone is a ~30 cm cube, so its apparent width is 0.30 m face-on and up to
# 0.42 m across a diagonal. At ZOOM_NORM=0.30 (fx = 2638 px):
#     30 px  ->  26.4 m (face-on) .. 36.9 m (diagonal)
#     40 px  ->  19.8 m (face-on) .. 27.7 m (diagonal)
# so this band lands roughly in the requested 30-40 px. Exact pixel size is not
# measurable without fixing full_drone's bbox, so treat it as approximate.
RANGE_M = (21.0, 33.0)

# Keep the drone this far from the frame edge so it is never clipped.
EDGE_MARGIN_PX = 90

# Random yaw gives varied silhouettes (and naturally varies apparent size).
RANDOM_DRONE_YAW = True

# ---- birds ------------------------------------------------------------------
# A single GIF frame is composited onto the SAVED PNG, after capture. Doing it
# here rather than in the sim is deliberate: we place it, so we know its exact
# pixel box, and it costs nothing in the render loop.
BIRD_PROBABILITY = 0.5          # fraction of images that get a bird
BIRD_PX = (24, 64)              # longer-side size range, px
BIRD_MIN_GAP_PX = 25            # keep this clear of the drone so labels are
                                # unambiguous (no overlapping targets)
GIFS_DIR = os.path.join(_ROOT, "gifs")

# ---- timing / robustness ----------------------------------------------------
SETTLE_TOLERANCE_M = 2.0        # reported vs commanded offset agreement
SETTLE_TIMEOUT_S = 4.0          # give up on a placement after this
CAPTURE_TIMEOUT_S = 15.0        # wait for the PNG to appear on disk
POST_TELEPORT_MIN_S = 0.35      # let the render catch up before capturing
SEED = 42                       # RNG seed: same seed => same dataset. Set to
                                # None for a different dataset every run.

HOST_ADDR = ("127.0.0.1", 5005)         # ptz_sim's command port
BBOX_TOPIC = "/isaac_core/bbox"
SAT_TOPIC = "/isaac_core/sat"
DRONE_TARGET = "Cube"                   # the marker we read pixels from

# Camera azimuth mapping, verified empirically against scene playback:
# OnePointSender(yaw=90) -> NED->ENU yaw 0 -> gimbal -pan*180 -> bearing below.
CAM_AZIMUTH_AT_PAN0_DEG = 90.0
CAM_AZIMUTH_PER_PAN_DEG = 180.0


# ==============================================================================
# geometry
# ==============================================================================

def _fx_px(hfov_deg, width=IMAGE_W):
    """Pinhole focal length in PIXELS for a given horizontal FoV."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def enu_offset_for_pixel(u, v, range_m, pan, tilt, hfov_deg):
    """ENU offset (east, north, up) placing a point at pixel (u, v), `range_m` away.

    Standard pinhole ray, built on a camera basis in ENU:
        forward from the pan/tilt angles, right horizontal, up = right x forward.
    `range_m` is true 3-D distance, so the value Isaac reports back should match.
    """
    fx = _fx_px(hfov_deg)
    dx = u - IMAGE_W / 2.0                  # +right
    dy = IMAGE_H / 2.0 - v                  # +up (image v grows downward)

    az = math.radians(CAM_AZIMUTH_AT_PAN0_DEG + pan * CAM_AZIMUTH_PER_PAN_DEG)
    el = math.radians(tilt * cam_model.TILT_DEG_PER_UNIT)

    fwd = (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))
    right = (math.cos(az), -math.sin(az), 0.0)
    up = (right[1] * fwd[2] - right[2] * fwd[1],     # right x forward
          right[2] * fwd[0] - right[0] * fwd[2],
          right[0] * fwd[1] - right[1] * fwd[0])

    d = [fwd[i] + (dx / fx) * right[i] + (dy / fx) * up[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in d)) or 1.0
    return tuple(c / n * range_m for c in d)


# ==============================================================================
# the running simulator, as seen from here
# ==============================================================================

class SimLink:
    """ROS subscriptions + the UDP command socket to ptz_sim."""

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("dataset_generator")

        # BEST_EFFORT is mandatory: bbox_node publishes with
        # qos_profile_sensor_data, and a RELIABLE subscriber cannot receive from a
        # BEST_EFFORT publisher — DDS delivers nothing while `ros2 topic echo`
        # still shows traffic, because echo adapts.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self._lock = threading.Lock()
        self._targets = {}          # name -> dict of the latest fields
        self._bbox_seq = 0
        self.node.create_subscription(FrameBboxes, BBOX_TOPIC, self._on_bbox, qos)
        self._sat_pub = self.node.create_publisher(SATOutput, SAT_TOPIC, 10)

        self._spinning = True
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.3)

    def _spin(self):
        """Cooperative spin.

        rclpy.spin() cannot be interrupted from outside, so the thread was still
        inside C++ when the interpreter tore down — the process then died with
        "terminate called without an active exception" AFTER all the work was
        safely on disk, which made the exit code meaningless. spin_once in a loop
        with a flag lets close() join the thread first.
        """
        while self._spinning and rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except Exception:
                break

    def _on_bbox(self, msg):
        seen = {}
        for b in msg.bboxes:
            seen[b.target_name] = {
                "in_frame": bool(b.in_frame),
                "is_visible": bool(b.is_visible),
                "x1": int(b.x1), "y1": int(b.y1), "x2": int(b.x2), "y2": int(b.y2),
                "east": float(b.distance_x),
                "north": float(b.distance_y),
                "up": float(b.distance_z),
            }
        with self._lock:
            self._targets = seen
            self._bbox_seq += 1

    def target(self, name=DRONE_TARGET):
        with self._lock:
            t = self._targets.get(name)
            return (dict(t) if t else None), self._bbox_seq

    def bbox_seq(self):
        with self._lock:
            return self._bbox_seq

    # ---- commands to ptz_sim ------------------------------------------------
    def send(self, msg_type, **kw):
        packet = {"type": msg_type, "time": time.time(), **kw}
        self.sock.sendto(json.dumps(packet).encode("utf-8"), HOST_ADDR)

    def wait_for_pose(self, pan, tilt, zoom_norm, timeout=12.0, tol=0.02):
        """Block until ptz_sim reports the requested pose. Returns the pose dict.

        Zoom slews over ~4 s in the sim, and pan/tilt snap to a ~0.02 grid, so
        this must be a real wait — starting to capture early would mean labels
        recorded at a field of view the image does not have.
        """
        end = time.time() + timeout
        last = None
        while time.time() < end:
            self.send("ping", session="dataset")      # keeps pose replies coming
            try:
                data, _ = self.sock.recvfrom(8192)
            except socket.timeout:
                continue
            try:
                m = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if m.get("type") != "pose":
                continue
            last = m
            z = cam_model.normalize_zoom(float(m["zoom"]))
            if (abs(float(m["pan"]) - pan) <= tol
                    and abs(float(m["tilt"]) - tilt) <= tol
                    and abs(z - zoom_norm) <= 0.01):
                return m
        return last

    def capture(self, path):
        """Ask Isaac to write the current viewport to `path`; wait for the file.

        FileCapture is asynchronous, so the publish returning tells us nothing —
        we must wait for the file. sat_node also ignores a repeated path, which is
        fine because every sample uses a fresh filename.
        """
        if os.path.exists(path):
            os.remove(path)
        msg = SATOutput()
        msg.output_path = path
        self._sat_pub.publish(msg)

        end = time.time() + CAPTURE_TIMEOUT_S
        last_size = -1
        while time.time() < end:
            if os.path.exists(path):
                size = os.path.getsize(path)
                # wait for the size to stop changing: the file appears before it
                # is fully written
                if size > 0 and size == last_size:
                    return True
                last_size = size
            time.sleep(0.1)
        return False

    def close(self):
        """Shut ROS down properly.

        Without the rclpy.shutdown() the spin thread is still inside spin() when
        the interpreter tears down, and the process dies with
        "terminate called without an active exception" / SIGABRT after all the
        work is already on disk — alarming, and it makes the exit code useless.
        """
        self._spinning = False
        try:
            self._spin_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


class DronePlacer:
    """Teleports the drone by driving /bboxes over UDP 33335."""

    def __init__(self, cam_lat, cam_lon, cam_alt):
        self.cam = [cam_lat, cam_lon, cam_alt]
        self.bot = UdpBot(
            udp_port=scenes.DRONE_UDP_PORT, send_rate_hz=30.0,
            start_lat=cam_lat, start_lon=cam_lon, start_alt=cam_alt + 50.0,
            start_roll_d=0.0, start_pitch_d=0.0, start_yaw_d=0.0,
        )
        # UdpBot's own loop re-publishes whatever pose is set, at send_rate_hz.
        # We only set the pose; publishing here as well would double the packet
        # rate into a receiver that consumes one packet per rendered frame.
        self.bot.run(blocking=False)

    def place(self, east, north, up, yaw_deg=None):
        """Move the drone to an ENU offset (metres) from the camera."""
        d_lat, d_lon = meters_to_latlon_offset(north, east, self.cam[0])
        self.bot._current_lat = self.cam[0] + d_lat
        self.bot._current_lon = self.cam[1] + d_lon
        self.bot._current_alt = self.cam[2] + up
        if yaw_deg is not None:
            # Mirror what UdpBot.__init__ does, so body angles stay consistent.
            from transforms3d.euler import euler2mat
            self.bot._world_rotation_matrix = euler2mat(
                0.0, 0.0, math.radians(yaw_deg), axes="rxyz")
            self.bot._update_cur_rotation_from_world_rotation_matrix()

    def close(self):
        try:
            self.bot.stop()
            time.sleep(0.15)        # let its sender loop exit before the socket goes
            self.bot.close()
        except Exception:
            pass


class BirdStamper:
    """Composites one GIF frame onto a saved PNG and reports where it landed."""

    def __init__(self, gifs_dir=GIFS_DIR):
        from PIL import Image, ImageSequence
        self._Image = Image
        self.frames = []            # (name, RGBA Image)
        if not os.path.isdir(gifs_dir):
            print(f"[gen] WARNING: no gifs dir at {gifs_dir} — birds disabled")
            return
        for name in sorted(os.listdir(gifs_dir)):
            if not name.lower().endswith(".gif"):
                continue
            try:
                im = Image.open(os.path.join(gifs_dir, name))
                for fr in ImageSequence.Iterator(im):
                    self.frames.append((name, fr.convert("RGBA").copy()))
            except Exception as e:
                print(f"[gen] WARNING: could not read {name}: {e}")
        print(f"[gen] birds: {len(self.frames)} frames from {gifs_dir}")

    def available(self):
        return bool(self.frames)

    def stamp(self, png_path, avoid_box, rng):
        """Paste one bird, avoiding `avoid_box`. Returns a label dict or None."""
        if not self.frames:
            return None
        name, src = rng.choice(self.frames)
        target = rng.randint(*BIRD_PX)
        w0, h0 = src.size
        scale = target / float(max(w0, h0))
        bw, bh = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
        bird = src.resize((bw, bh), self._Image.LANCZOS)
        if rng.random() < 0.5:
            bird = bird.transpose(self._Image.FLIP_LEFT_RIGHT)

        ax1, ay1, ax2, ay2 = avoid_box
        for _ in range(60):                     # a few tries to miss the drone
            x = rng.randint(0, IMAGE_W - bw)
            y = rng.randint(0, IMAGE_H - bh)
            g = BIRD_MIN_GAP_PX
            if (x + bw + g < ax1 or x - g > ax2
                    or y + bh + g < ay1 or y - g > ay2):
                break
        else:
            return None                         # frame too crowded; skip

        base = self._Image.open(png_path).convert("RGBA")
        base.alpha_composite(bird, dest=(x, y))
        base.convert("RGB").save(png_path)
        return {
            "present": True,
            "center_px": [x + bw / 2.0, y + bh / 2.0],
            "bbox_px": [x, y, x + bw, y + bh],
            "size_px": [bw, bh],
            "source": name,
        }


NO_BIRD = {"present": False, "center_px": None, "bbox_px": None,
           "size_px": None, "source": None}


# ==============================================================================
# JSON
# ==============================================================================

FIELD_DOCS = {
    "_about": "One sample of the drone-on-sky dataset, generated by "
              "ptz_sim/scripts/gen_dataset.py. Every field below is read back "
              "from the simulator after the drone was placed; nothing is the "
              "generator's own estimate.",
    "image": "PNG filename next to this JSON.",
    "image_width/height": "Pixel dimensions of that PNG. Pixel labels are in "
                          "this coordinate space, origin top-left.",
    "drone.center_px": "[x, y] centre of the drone in pixels. Midpoint of the "
                       "'Cube' marker's 2-D box from /isaac_core/bbox. Cube is "
                       "an 8 cm mesh rigidly attached to the drone (both move "
                       "with their shared parent prim), offset under a pixel at "
                       "these ranges.",
    "drone.bbox_px": "[x1, y1, x2, y2] of the Cube marker itself, NOT the "
                     "drone's silhouette. It is a ~1 px box; use center_px. The "
                     "drone's own box is unavailable (its semantic tags sit on "
                     "child meshes, so the bbox publisher reports -1 for it).",
    "drone.range_m": "Straight-line camera-to-drone distance in metres.",
    "drone.offset_m": "Camera-relative offset in metres, ENU (east/north/up).",
    "drone.approx_size_px": "Rough on-screen drone width, from its ~0.3 m size, "
                            "the range and the focal length. An ESTIMATE — the "
                            "true rendered size is not measurable here.",
    "camera.zoom_norm": "Zoom as 0.0 (fully wide) .. 1.0 (fully tele).",
    "camera.zoom_raw": "The same zoom in the camera's raw ONVIF units (-1..+1), "
                       "which is what the hardware reports.",
    "camera.focal_length_mm": "Focal length the sim applied for this zoom.",
    "camera.hfov_deg": "Horizontal field of view at this zoom.",
    "camera.pan/tilt": "Camera pose in ONVIF units (-1..+1). tilt>0 looks up; "
                       "1.0 tilt = 45 deg.",
    "bird.present": "Whether a bird was composited into this image. The other "
                    "bird fields are null when false — the keys are ALWAYS "
                    "present so downstream code can read them unconditionally.",
    "bird.center_px": "[x, y] centre of the bird in pixels.",
    "bird.bbox_px": "[x1, y1, x2, y2] of the bird sprite (exact: we pasted it).",
    "bird.size_px": "[width, height] of the pasted sprite.",
    "bird.source": "Which GIF the frame came from.",
    "meta.seed": "RNG seed for the whole run. Re-running with the same seed "
                 "reproduces the same dataset. Set SEED=None for a fresh one.",
}


def write_json(path, sample):
    with open(path, "w") as fh:
        json.dump({"_fields": FIELD_DOCS, **sample}, fh, indent=2)


# ==============================================================================
# main
# ==============================================================================

def calibrate_camera_origin(link, placer, hfov_deg, pan, tilt):
    """Correct the assumed camera position using one measured placement.

    scenes.CAMERA is where ptz_sim puts the camera, but if that ever drifts from
    reality every shot would miss by the same amount and the run would look
    mysteriously broken. One placement measures the error and removes it:

        commanded = assumed_cam + desired          (we chose this)
        reported  = commanded - true_cam           (Isaac tells us)
        => true_cam = assumed_cam + (desired - reported)
    """
    desired = enu_offset_for_pixel(IMAGE_W / 2.0, IMAGE_H / 2.0,
                                   sum(RANGE_M) / 2.0, pan, tilt, hfov_deg)
    placer.place(*desired, yaw_deg=0.0)
    seq0 = link.bbox_seq()
    end = time.time() + SETTLE_TIMEOUT_S
    got = None
    while time.time() < end:
        t, seq = link.target()
        if t and seq > seq0 + 2:
            got = t
            break
        time.sleep(0.05)
    if got is None:
        print("[gen] WARNING: no bbox telemetry during calibration — is "
              "data_sim.py running with bbox_publisher=True?")
        return (0.0, 0.0, 0.0)
    err = (desired[0] - got["east"], desired[1] - got["north"],
           desired[2] - got["up"])
    dist = math.sqrt(sum(c * c for c in err))
    print(f"[gen] camera-origin calibration: correction "
          f"E{err[0]:+.2f} N{err[1]:+.2f} U{err[2]:+.2f} m  (|{dist:.2f}| m)")
    if dist > 500.0:
        print("[gen] NOTE: that is a large correction — the assumed camera "
              "position was well off, but it is now compensated.")
    return err


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    hfov = cam_model.hfov_deg(ZOOM_NORM)
    focal = cam_model.focal_length_mm(ZOOM_NORM)
    zoom_raw = cam_model.denormalize_zoom(ZOOM_NORM)
    fx = _fx_px(hfov)
    print(f"[gen] zoom {ZOOM_NORM} -> HFoV {hfov:.2f} deg, focal {focal:.2f} mm, "
          f"fx {fx:.0f} px")
    print(f"[gen] tilt {TILT_UNITS} -> {TILT_UNITS * cam_model.TILT_DEG_PER_UNIT:.1f} "
          f"deg up; range band {RANGE_M[0]}-{RANGE_M[1]} m")
    print(f"[gen] writing {NUM_IMAGES} samples to {OUT_DIR}")

    link = SimLink()
    birds = BirdStamper()

    # Camera pose first: pan is randomised per sample, so start centred.
    pan0 = 0.0
    link.send("absolute_move", pan=pan0, tilt=TILT_UNITS, zoom=zoom_raw)
    pose = link.wait_for_pose(pan0, TILT_UNITS, ZOOM_NORM)
    if pose is None:
        print("[gen] FATAL: no pose replies from ptz_sim on "
              f"{HOST_ADDR[0]}:{HOST_ADDR[1]} — is data_sim.py running?")
        return 1
    print(f"[gen] camera settled: pan={pose['pan']:.3f} tilt={pose['tilt']:.3f} "
          f"zoom_raw={pose['zoom']:.3f}")

    placer = DronePlacer(scenes.CAMERA["lat"], scenes.CAMERA["lon"],
                         scenes.CAMERA["alt"])
    corr = calibrate_camera_origin(link, placer, hfov, pose["pan"], pose["tilt"])
    placer.cam[0] += meters_to_latlon_offset(corr[1], corr[0], placer.cam[0])[0]
    placer.cam[1] += meters_to_latlon_offset(corr[1], corr[0], placer.cam[0])[1]
    placer.cam[2] += corr[2]

    made = attempts = 0
    t_start = time.time()
    while made < NUM_IMAGES:
        attempts += 1
        if attempts > NUM_IMAGES * 6 + 50:
            print("[gen] giving up: too many rejected placements")
            break

        # ---- choose a target pixel, range, pose ----------------------------
        pan = round(rng.uniform(*PAN_RANGE) / 0.02) * 0.02      # honour the grid
        u = rng.uniform(EDGE_MARGIN_PX, IMAGE_W - EDGE_MARGIN_PX)
        v = rng.uniform(EDGE_MARGIN_PX, IMAGE_H - EDGE_MARGIN_PX)
        rng_m = rng.uniform(*RANGE_M)
        yaw = rng.uniform(0.0, 360.0) if RANDOM_DRONE_YAW else 0.0

        if abs(pan - pose["pan"]) > 1e-9:
            link.send("absolute_move", pan=pan, tilt=TILT_UNITS)
            pose = link.wait_for_pose(pan, TILT_UNITS, ZOOM_NORM, timeout=6.0)
            if pose is None:
                print("[gen] lost pose replies; aborting")
                break

        desired = enu_offset_for_pixel(u, v, rng_m, pose["pan"], pose["tilt"], hfov)
        placer.place(*desired, yaw_deg=yaw)

        # ---- wait for the sim to actually reflect it ------------------------
        seq0 = link.bbox_seq()
        deadline = time.time() + SETTLE_TIMEOUT_S
        t_min = time.time() + POST_TELEPORT_MIN_S
        tgt = None
        while time.time() < deadline:
            cand, seq = link.target()
            if cand and seq > seq0 + 1 and time.time() >= t_min:
                err = math.dist((cand["east"], cand["north"], cand["up"]), desired)
                if err <= SETTLE_TOLERANCE_M:
                    tgt = cand
                    break
            time.sleep(0.05)

        if tgt is None:
            continue                            # placement never converged
        if not tgt["in_frame"]:
            continue                            # off camera; try another spot

        cx = (tgt["x1"] + tgt["x2"]) / 2.0
        cy = (tgt["y1"] + tgt["y2"]) / 2.0
        if not (0 <= cx < IMAGE_W and 0 <= cy < IMAGE_H):
            continue                            # in_frame but no usable box

        # ---- capture -------------------------------------------------------
        stem = f"{made:06d}"
        png = os.path.join(OUT_DIR, stem + ".png")
        if not link.capture(png):
            print(f"[gen] capture timed out for {stem}; is sat=True in the "
                  f"launcher (scripts/data_sim.py)?")
            continue

        # ---- optional bird -------------------------------------------------
        rng_used = math.sqrt(tgt["east"] ** 2 + tgt["north"] ** 2 + tgt["up"] ** 2)
        approx_px = 0.30 * fx / max(1e-6, rng_used)
        half = max(8.0, approx_px)
        drone_box = (cx - half, cy - half, cx + half, cy + half)
        bird = NO_BIRD
        if birds.available() and rng.random() < BIRD_PROBABILITY:
            bird = birds.stamp(png, drone_box, rng) or NO_BIRD

        # ---- labels --------------------------------------------------------
        write_json(os.path.join(OUT_DIR, stem + ".json"), {
            "image": stem + ".png",
            "image_width": IMAGE_W,
            "image_height": IMAGE_H,
            "drone": {
                "center_px": [cx, cy],
                "bbox_px": [tgt["x1"], tgt["y1"], tgt["x2"], tgt["y2"]],
                "range_m": rng_used,
                "offset_m": {"east": tgt["east"], "north": tgt["north"],
                             "up": tgt["up"]},
                "approx_size_px": approx_px,
            },
            "camera": {
                "zoom_norm": ZOOM_NORM,
                "zoom_raw": float(pose["zoom"]),
                "focal_length_mm": focal,
                "hfov_deg": hfov,
                "pan": float(pose["pan"]),
                "tilt": float(pose["tilt"]),
            },
            "bird": bird,
            "meta": {"seed": SEED, "timestamp": time.time()},
        })

        made += 1
        if made % 10 == 0 or made == 1:
            rate = made / max(1e-6, time.time() - t_start)
            print(f"[gen] {made}/{NUM_IMAGES}  ({rate:.2f} img/s, "
                  f"{attempts} attempts)  last: px=({cx:.0f},{cy:.0f}) "
                  f"range={rng_used:.1f}m ~{approx_px:.0f}px "
                  f"bird={'yes' if bird['present'] else 'no'}")

    dt = time.time() - t_start
    print(f"[gen] done: {made} samples in {dt:.0f}s "
          f"({made / max(1e-6, dt):.2f} img/s, {attempts} attempts) -> {OUT_DIR}")
    placer.close()
    link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
