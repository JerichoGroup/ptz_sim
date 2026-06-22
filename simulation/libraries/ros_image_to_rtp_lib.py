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

import math
import os
import random
import threading
import time
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GstRtspServer

from libraries.sim_lib import SimLibBase


# ==================== Frame effects (blur + noise) ====================
# Optional realism layer applied to each frame *before* it is H.264-encoded:
#   - zoom-refocus BLUR: a Gaussian blur that ramps up then clears after a zoom,
#     mimicking the real lens hunting focus when it changes magnification.
#   - sensor NOISE: small dark drifting specks (dust / flies / wind debris) that
#     move coherently across the frame.
#
# Everything is tunable via the EFFECTS instance below. Requires numpy + OpenCV;
# if either is missing the layer disables itself and frames pass through clean.

try:
    import numpy as np
    import cv2
    _CV_OK = True
except Exception as _cv_err:                              # pragma: no cover
    _CV_OK = False
    print(f"[FrameEffects] numpy/cv2 unavailable ({_cv_err}); effects disabled")


# Sprite sources live in <repo>/pics (static PNGs) and <repo>/gifs (animations);
# resolve relative so it works on any host.
_PICS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "pics"))
_GIFS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "gifs"))


def _pics(*names):
    return [os.path.join(_PICS_DIR, n) for n in names]


def _gifs(*names):
    return [os.path.join(_GIFS_DIR, n) for n in names]


@dataclass
class SpriteKind:
    """One family of drifting objects (e.g. flies, birds).

    A random source from `images` is chosen per spawn and scaled (aspect-ratio
    preserved) so its LONGER side equals a random value in [size_min, size_max].
    Sources may be static PNGs *or* animated GIFs — a GIF's frames are played
    back (one gif-frame held for `anim_hold` rendered frames) so the sprite
    actually moves (e.g. birds flapping). `count` is the max on screen at once;
    `gap_*` idle frames between a slot's appearances make the on-screen count
    fluctuate down to 0 — bigger gaps ⇒ rarer ⇒ lower "rate".
    """
    name:      str
    images:    list                       # PNG and/or GIF paths; [] → gray squares
    count:     int   = 1                  # max simultaneously on screen
    size_min:  int   = 6                  # longer-side length in px
    size_max:  int   = 20
    speed_min: float = 4.0                # px/frame
    speed_max: float = 40.0
    life_min:  int   = 25                 # frames present before it leaves
    life_max:  int   = 120
    gap_min:   int   = 20                 # frames absent before reappearing
    gap_max:   int   = 160
    alpha:     float = 0.9                # 0..1 opacity multiplier
    jitter:    float = 0.6                # per-frame velocity wander
    anim_hold: int   = 3                  # rendered frames each GIF frame is shown
    gray:      int   = 25                 # fallback square shade if images missing


def _default_kinds():
    return [
        # Flies: small, fast, frequent (0–2 on screen). Static PNGs (too small
        # for animation to read on-screen).
        SpriteKind(
            name="fly", images=_pics("fly1.png", "fly2.png", "fly3.png"),
            count=2, size_min=3, size_max=14,
            speed_min=10.0, speed_max=70.0,
            life_min=25, life_max=120, gap_min=8, gap_max=70,
            alpha=0.85, jitter=0.9),
        # Birds: smaller-than-before, slow, rare (0–1, ~half the fly rate),
        # animated from flapping-bird GIFs.
        SpriteKind(
            name="bird",
            images=_gifs("bird1.gif", "bird2.gif", "bird3.gif", "bird4.gif", "bird5.gif"),
            count=1, size_min=14, size_max=40,
            speed_min=2.0, speed_max=12.0,
            life_min=70, life_max=240, gap_min=90, gap_max=260,
            alpha=0.92, jitter=0.35, anim_hold=3),
    ]


@dataclass
class FrameEffectsConfig:
    """Edit the EFFECTS instance below to configure blur and noise."""

    # ---- zoom-refocus blur ----
    blur_enabled:    bool  = True
    max_blur:        float = 1.0    # 0..1 overall blur strength (scales blur_max_sigma)
    blur_time:       float = 2.0    # s: total refocus settle window after a zoom
    blur_peak_frac:  float = 0.4    # fraction of blur_time where blur peaks (≈0.8 s)
    blur_max_sigma:  float = 9.0    # Gaussian sigma (px) at full intensity; ↑ = blurrier
    blur_rearm_gap:  float = 0.3    # s gap between zoom changes that starts a NEW blur
    zoom_topic:      str   = "/isaac_core/zoom"
    zoom_epsilon:    float = 1e-4   # min zoom delta counted as "a zoom happened"

    # ---- sensor noise (drifting sprites: flies, birds, …) ----
    noise_enabled:   bool  = True
    kinds:           list  = field(default_factory=_default_kinds)


EFFECTS = FrameEffectsConfig()      # ←—— configure blur / noise here


class FrameEffects:
    """Applies zoom-refocus blur and drifting-sprite noise to raw frame bytes.

    All methods run on the single ROS spin thread (the image and zoom callbacks
    are serialised by rclpy's single-threaded executor), so no locking is needed.
    Any failure falls back to returning the frame untouched — streaming never
    breaks because of an effect.
    """

    def __init__(self, cfg: FrameEffectsConfig) -> None:
        self.cfg = cfg
        self.enabled = _CV_OK
        # blur state
        self._zoom_last_val = None
        self._blur_t0 = None          # monotonic start of the current blur envelope
        self._zoom_last_t = 0.0       # monotonic time of the last zoom change
        # noise state — one entry per kind:
        #   {"cfg", "bases":[ [ (color,alpha), ... ] ], "particles":[]}
        # each base is an animation: a list of (color, alpha) frames (1 for PNGs).
        self._kind_states = []
        self._sprites_loaded = False
        self._noise_dims = None       # (w, h) the particles were seeded for

    # ---- zoom signal → arms the blur envelope ----
    def notify_zoom(self, value: float) -> None:
        if not (self.enabled and self.cfg.blur_enabled):
            return
        if self._zoom_last_val is None:
            self._zoom_last_val = value
            return
        if abs(value - self._zoom_last_val) > self.cfg.zoom_epsilon:
            now = time.monotonic()
            # Anchor the envelope to the START of a zoom gesture so blur builds
            # while zooming and peaks shortly after. Re-arm when the previous
            # envelope has finished (long continuous slew) or after an idle gap
            # (a distinct new zoom command).
            if (self._blur_t0 is None
                    or (now - self._blur_t0) >= self.cfg.blur_time
                    or (now - self._zoom_last_t) > self.cfg.blur_rearm_gap):
                self._blur_t0 = now
            self._zoom_last_t = now
        self._zoom_last_val = value

    # ---- blur envelope: 0 → peak (at blur_peak_frac) → 0 (at blur_time) ----
    def _blur_sigma(self, now: float) -> float:
        if self._blur_t0 is None:
            return 0.0
        tau = now - self._blur_t0
        bt = self.cfg.blur_time
        if tau < 0 or tau >= bt:
            return 0.0
        p = tau / bt
        pk = self.cfg.blur_peak_frac
        a = p / pk if p <= pk else (1.0 - p) / (1.0 - pk)   # triangle 0..1
        a = max(0.0, min(1.0, a))
        a = a * a * (3.0 - 2.0 * a)                          # smoothstep (soft)
        return a * self.cfg.max_blur * self.cfg.blur_max_sigma

    # ---- sprite sources (loaded once, in the frame's channel order) ----
    @staticmethod
    def _load_source(path, order):
        """Load a PNG (1 frame) or GIF (N frames) → list of (color, alpha) frames
        in the frame's channel order, color float32, alpha float32 0..1."""
        frames = []
        if os.path.splitext(path)[1].lower() == ".gif":
            from PIL import Image, ImageSequence
            im = Image.open(path)
            for fr in ImageSequence.Iterator(im):
                arr = np.asarray(fr.convert("RGBA"))      # PIL → RGBA (RGB order)
                color = arr[:, :, :3].astype(np.float32)
                alpha = arr[:, :, 3].astype(np.float32) / 255.0
                if order == "bgr":                        # frame wants BGR
                    color = color[:, :, ::-1].copy()
                frames.append((color, alpha))
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2 → BGR(A)
            if img is None:
                raise RuntimeError("imread returned None")
            if img.ndim == 3 and img.shape[2] == 4:
                color = img[:, :, :3].astype(np.float32)
                alpha = img[:, :, 3].astype(np.float32) / 255.0
            elif img.ndim == 3:
                color = img[:, :, :3].astype(np.float32)
                alpha = np.ones(img.shape[:2], np.float32)
            else:
                color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
                alpha = np.ones(img.shape[:2], np.float32)
            if order == "rgb":                            # frame wants RGB
                color = color[:, :, ::-1].copy()
            frames.append((color, alpha))
        return frames

    def _load_sprites(self, order: str) -> None:
        if self._sprites_loaded:
            return
        self._sprites_loaded = True
        self._kind_states = []
        for k in self.cfg.kinds:
            bases = []                       # each base is a list of (color, alpha) frames
            for path in k.images:
                try:
                    bases.append(self._load_source(path, order))
                except Exception as e:
                    print(f"[FrameEffects] {k.name}: could not load {path} ({e})")
            if not bases:
                print(f"[FrameEffects] {k.name}: no sources → gray-square fallback")
            self._kind_states.append({"cfg": k, "bases": bases, "particles": []})
        print("[FrameEffects] sprites: " + ", ".join(
            f"{ks['cfg'].name}×{len(ks['bases'])}" for ks in self._kind_states))

    def _make_anim(self, base_frames, size, alpha_mul):
        """Scale every frame of an animation so its longer side == `size` (aspect
        preserved), with one random L/R flip applied to all frames so the sprite
        keeps a consistent heading. Returns (frames, sw, sh)."""
        h0, w0 = base_frames[0][1].shape[:2]
        scale = size / float(max(h0, w0))
        nw = max(1, int(round(w0 * scale)))
        nh = max(1, int(round(h0 * scale)))
        flip = random.random() < 0.5
        out = []
        for color, alpha in base_frames:
            c = cv2.resize(color, (nw, nh), interpolation=cv2.INTER_AREA)
            a = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
            if flip:
                c = cv2.flip(c, 1)
                a = cv2.flip(a, 1)
            out.append((c, a * alpha_mul))
        return out, nw, nh

    # ---- noise particles ----
    def _spawn(self, ks, w, h):
        """A freshly-active sprite of kind `ks` (wait=0 → drawn immediately)."""
        k = ks["cfg"]
        speed = random.uniform(k.speed_min, k.speed_max)
        ang = random.uniform(0.0, 2.0 * math.pi)
        size = random.randint(k.size_min, k.size_max)
        if ks["bases"]:
            frames, sw, sh = self._make_anim(random.choice(ks["bases"]), size, k.alpha)
        else:
            frames = None
            sw = sh = size
        return {
            "x": random.uniform(0, w), "y": random.uniform(0, h),
            "vx": speed * math.cos(ang), "vy": speed * math.sin(ang),
            "sw": sw, "sh": sh, "gray": k.gray,
            "life": random.randint(k.life_min, k.life_max),
            "wait": 0,
            "frames": frames,                       # list of (color, alpha) or None
            "fidx": 0,                              # current animation frame
            "fhold": k.anim_hold,                   # ticks left on the current frame
        }

    def _ensure_particles(self, w, h):
        if self._noise_dims == (w, h):
            return
        for ks in self._kind_states:
            k = ks["cfg"]
            parts = []
            for _ in range(k.count):
                p = self._spawn(ks, w, h)
                # Stagger starts with a random initial gap so they don't all
                # appear together (scene often begins with 0–1 visible).
                p["wait"] = random.randint(0, k.gap_max)
                parts.append(p)
            ks["particles"] = parts
        self._noise_dims = (w, h)

    def _draw_noise(self, frame, w, h, ch):
        for ks in self._kind_states:
            k = ks["cfg"]
            lo, hi = k.speed_min, k.speed_max
            for p in ks["particles"]:
                # Inactive (between appearances): count down, then respawn.
                if p["wait"] > 0:
                    p["wait"] -= 1
                    if p["wait"] == 0:
                        p.update(self._spawn(ks, w, h))
                    continue
                # drift with a small velocity wander for organic motion
                p["vx"] += random.uniform(-k.jitter, k.jitter)
                p["vy"] += random.uniform(-k.jitter, k.jitter)
                sp = math.hypot(p["vx"], p["vy"]) or 1.0
                if sp > hi:
                    p["vx"] *= hi / sp; p["vy"] *= hi / sp
                elif sp < lo:
                    p["vx"] *= lo / sp; p["vy"] *= lo / sp
                p["x"] += p["vx"]; p["y"] += p["vy"]; p["life"] -= 1
                # when it ages out or drifts off-frame, go idle for a random gap
                sw, sh = p["sw"], p["sh"]
                m = max(sw, sh) + 2
                if (p["life"] <= 0 or p["x"] < -m or p["x"] > w + m
                        or p["y"] < -m or p["y"] > h + m):
                    p["wait"] = random.randint(k.gap_min, k.gap_max)
                    continue
                # advance the wing-flap animation (no-op for 1-frame sprites)
                frames = p["frames"]
                if frames is not None and len(frames) > 1:
                    p["fhold"] -= 1
                    if p["fhold"] <= 0:
                        p["fidx"] = (p["fidx"] + 1) % len(frames)
                        p["fhold"] = k.anim_hold
                # visible rect on the frame, and matching sub-rect of the sprite
                px = int(p["x"]); py = int(p["y"])
                x0 = max(0, px); y0 = max(0, py)
                x1 = min(w, px + sw); y1 = min(h, py + sh)
                if x1 <= x0 or y1 <= y0:
                    continue
                if frames is not None and ch >= 3:
                    spr_color, spr_alpha = frames[p["fidx"]]
                    sx0 = x0 - px; sy0 = y0 - py
                    spr = spr_color[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
                    al = spr_alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][..., None]
                    roi = frame[y0:y1, x0:x1, :3]
                    roi[:] = (roi * (1.0 - al) + spr * al).astype(np.uint8)
                else:
                    a = k.alpha
                    roi = frame[y0:y1, x0:x1]
                    roi[:] = (roi * (1.0 - a) + p["gray"] * a).astype(np.uint8)

    # ---- main entry: returns possibly-modified raw frame bytes ----
    def process(self, raw: bytes, w: int, h: int, c: int, encoding: str) -> bytes:
        if not self.enabled:
            return raw
        cfg = self.cfg
        sigma = self._blur_sigma(time.monotonic()) if cfg.blur_enabled else 0.0
        do_blur = sigma > 0.3
        do_noise = cfg.noise_enabled and any(k.count > 0 for k in cfg.kinds)
        if not do_blur and not do_noise:
            return raw                       # fast path: nothing to do this frame
        try:
            if len(raw) != w * h * c:
                return raw                   # padded/odd layout — don't risk corruption
            arr = np.frombuffer(raw, np.uint8).reshape(h, w, c)
            frame = cv2.GaussianBlur(arr, (0, 0), sigma) if do_blur else arr.copy()
            if do_noise:
                order = "rgb" if encoding in ("rgb8", "rgba8") else "bgr"
                self._load_sprites(order)
                self._ensure_particles(w, h)
                self._draw_noise(frame, w, h, c)
            return frame.tobytes()
        except Exception as e:
            print(f"[FrameEffects] process failed, passing frame through: {e}")
            return raw


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
    # Queue is leaky=downstream: when the encoder is busy and a new frame arrives,
    # the stale waiting frame is dropped so the encoder always gets the freshest frame.
    # key-int-max=30: force an IDR every ~1 s so that any artifact from a skipped
    # reference frame clears up quickly (vs. the x264 default of 250 frames).
    # max-bytes=0 on appsrc: default is 200 KB, but raw 1080p frames are ~6 MB.
    # Without unlimited buffer, every push after the first would silently fail
    # because 6 MB > 200 KB, and block=false makes the failure non-blocking/silent.
    # The explicit "video/x-raw,format=I420" caps after videoconvert force 4:2:0
    # chroma. Without it, RGB input negotiates to 4:4:4 (Y444), which the H.264
    # baseline profile cannot encode — x264 then errors per-frame and emits
    # corrupt output (smearing / wrong colors). I420 is the universally-supported
    # 4:2:0 format every H.264 decoder expects.
    _FACTORY_PIPELINE = (
        "( appsrc name=src is-live=true do-timestamp=true block=false format=time max-bytes=0 "
        "! queue max-size-buffers=2 leaky=downstream "
        "! videoconvert "
        "! video/x-raw,format=I420 "
        "! x264enc tune=zerolatency speed-preset=veryfast bitrate=10000 key-int-max=30 "
        "! h264parse "
        "! rtph264pay config-interval=1 pt=96 name=pay0 )"
    )

    # ROS image encoding → channel count, for the FrameEffects numpy reshape.
    _CHANNELS = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}

    def __init__(self, rtsp_port: int, rtsp_paths: list) -> None:
        self._rtsp_port = rtsp_port
        self._rtsp_paths = rtsp_paths

        # Realism layer (zoom-refocus blur + drifting-speck noise).
        self.effects = FrameEffects(EFFECTS)

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
        factory.props.latency = 0  # remove the 200 ms server-side jitter buffer
        factory.connect("media-configure", self._on_media_configure)

        mounts = server.get_mount_points()
        for path in self._rtsp_paths:
            mounts.add_factory(path, factory)
            print(f"[RTSP] Mounted at {path}")

        server.attach(None)  # attaches to default GLib main context
        self._rtsp_server = server

    def _on_media_configure(self, factory, media) -> None:
        """Fired in GLib thread when the first RTSP client connects."""
        media.set_latency(0)  # belt-and-suspenders: also clear media-level jitter buffer
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

        # Apply blur/noise realism before encoding (no-op when nothing is active).
        channels = self._CHANNELS.get(msg.encoding)
        if channels is not None:
            raw = self.effects.process(raw, msg.width, msg.height, channels, msg.encoding)

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
        # BEST_EFFORT + depth=1: always deliver the newest frame; drop stale ones.
        # Default RELIABLE/depth=10 can queue up to 10 frames (~333 ms at 30 Hz).
        _qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Image, topic, self._callback, _qos)

        # Zoom feed drives the refocus blur. Default QoS (RELIABLE/10) matches
        # ptz_sim's zoom publisher. Same single-threaded executor as the image
        # callback, so FrameEffects state needs no locking.
        self.zoom_sub = self.create_subscription(
            Float32, EFFECTS.zoom_topic, self._zoom_callback, 10
        )
        self.get_logger().info(
            f"Subscribing to {topic} (+ zoom {EFFECTS.zoom_topic}), "
            f"RTSP server on :{rtsp_port}"
        )

    def _callback(self, msg: Image) -> None:
        self.bridge.send_image(msg)

    def _zoom_callback(self, msg: Float32) -> None:
        self.bridge.effects.notify_zoom(float(msg.data))

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
