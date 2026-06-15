"""Typed configuration dataclasses for the DroneTracker system.

Every field has a default that matches the current hard-coded constant in
motion_ptz_pipeline.py so the system works out-of-the-box without a config
file.  Override via config.yaml / .env (see loader.py).
"""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """ONVIF camera connection parameters.

    Never hard-code credentials in source.  Load them from .env or config.yaml:
        CAM_USER=admin
        CAM_PASS=<your-password>
    """
    ip:   str = "192.168.1.32"
    port: int = 80
    user: str = "admin"
    password: str = ""   # default empty; must be overridden in config.yaml/.env

    @property
    def rtsp_main(self) -> str:
        return (
            f"rtsp://{self.user}:{self.password}@{self.ip}:554"
            "/unicast/c1/s0/live"
        )

    @property
    def rtsp_alt(self) -> str:
        return (
            f"rtsp://{self.user}:{self.password}@{self.ip}:554"
            "/cam/realmonitor?channel=1&subtype=0"
        )


@dataclass
class ZoomConfig:
    """Optical-zoom control parameters."""
    search_pos:     float = 0.0    # widest FOV for IDLE scanning
    track_pos:      float = 0.60   # initial zoom when entering TRACK
    max_pos:        float = 1.0
    step:           float = 0.05
    full_travel_s:  float = 7.0    # UNV 45× lens: full range in ~7 s
    manual_speed:   float = 0.3    # ONVIF velocity for arrow-key pan/tilt
    zoom_in_px:     int   = 35     # blob px below which we zoom in
    zoom_out_px:    int   = 200    # blob px above which we zoom out
    cooldown_frames: int  = 20     # minimum frames between zoom commands


@dataclass
class PidConfig:
    """PID gains and limits for pan/tilt velocity commands."""
    kp:       float = 0.6    # pre-calibration fallback; RelativeMove takes over after calib
    ki:       float = 0.01
    kd:       float = 0.05
    max_out:  float = 1.0    # full ONVIF velocity; vel_scale not applied to PID output
    deadband: float = 0.04
    i_limit:  float = 0.3


@dataclass
class MotionConfig:
    """Motion detection and ground-mask parameters."""
    min_area:              float = 2.0
    max_area:              float = float("inf")   # backstop for sky-ghost blobs
    cluster_distance:      float = 80.0
    ground_mask_interval_s: float = 120.0         # idle periodic recompute
    # Region to blank before motion detection — suppresses burned-in OSD timestamps.
    # Format: [x, y, w, h] in pixels, or null/empty to disable.
    # Tune w/h to fit the actual timestamp size on screen.
    osd_mask_rect:         list  = field(default_factory=list)


@dataclass
class TrackerConfig:
    """Norfair tracker parameters."""
    distance_threshold:  float = 120.0
    hit_counter_max:     int   = 200    # survive full camera-move+settle blackout (~6 s at 30 fps)
    initialization_delay: int  = 2
    trail_max_age_s:     float = 2.0    # seconds before an unseen track's trail is deleted


@dataclass
class YoloConfig:
    """YOLO classification and auto-zoom parameters."""
    model_path:     str   = "results/runs/detect/train-15/weights/best.pt"
    imgsz:          int   = 640
    conf:           float = 0.18
    fill_frac:      float = 0.15    # target bbox fill fraction of frame after zoom
    edge_margin:    float = 0.05    # bbox centre fraction from edge that triggers zoom-out
    hold_timeout_s: float = 1.5    # consecutive YOLO misses before going home
    rel_move_max:   float = 0.15   # ONVIF unit clamp for delta_pan / delta_tilt
    # Must equal the --batch used when exporting the TensorRT engine (if using .engine).
    # 1920×1080 @640 px tiles → 8 (default); 2560×1440 → 15.
    warmup_tiles:   int   = 8


@dataclass
class TrackConfig:
    """TRACK-state control and focus parameters.

    NOTE: After the dead-code removal, most fields below are no longer used by
    the live pipeline — they were only referenced by the deleted _step_track /
    _track_acquire_phase / _track_yolo_phase methods.  ``min_vel_scale`` is
    still read by PTZController.  The rest are kept for config-file
    compatibility and potential future use.
    """
    lost_timeout_s:  float = 1.5   # (unused) seconds without locked track → return to IDLE
    center_tol:      float = 0.10  # (unused) normalised half-error to count as "centred"
    center_frames:   int   = 8     # (unused) consecutive centred frames before zoom-in
    focus_interval_s: float = 2.0  # (unused) seconds between autofocus triggers
    min_vel_scale:   float = 0.30  # floor for vel_scale() (arrow keys only) — still used


@dataclass
class FrozenConfig:
    """TRACK lock-on parameters (legacy config key: ``frozen``).

    Despite the class name this configures the TRACK state (formerly called
    FROZEN).  The config key remains ``frozen`` for backwards compatibility
    with existing config.yaml files.
    """
    fov_sign_x:     int   = +1    # flip if first live test moves the wrong way
    fov_sign_y:     int   = -1    # most likely axis to need flipping
    fov_gain:       float = 1.0   # try 0.5 if it overshoots ~2×
    move_settle_s:  float = 1.2   # gate: classify only after this many s from lock (pan/tilt settle)
    focus_settle_s: float = 1.5   # hard-cap: classify even if AF hasn't plateaued after this many s
    # Zoom-in on lock-on: doubles optical magnification by default.
    zoom_factor:   float = 4.0   # magnification multiplier applied at T/Enter lock
    zoom_parallel: bool  = True  # False → zoom starts after pan/tilt settles (sequential)
    # Sharpness-gated autofocus convergence detection (see _step_track Phase B).
    focus_plateau_frames: int   = 4     # consecutive non-improving Laplacian frames → AF converged
    focus_roi_frac:       float = 0.30  # center-crop fraction of min(h,w) for sharpness measurement
    # Post-classify monitoring throttle: run full-frame tiled YOLO at most this often.
    # Keeps detect-loop FPS high while still checking for sustained drone loss.
    monitor_interval_s:   float = 0.1   # min seconds between post-classify YOLO passes (~10×/sec)


@dataclass
class HomeConfig:
    """Return-to-home parameters."""
    kp:       float = 3.0
    tol:      float = 0.01
    timeout_s: float = 8.0
    max_vel:  float = 1.0


@dataclass
class PtzCommandConfig:
    """PTZ command coalescing parameters."""
    cmd_ttl: float = 0.15   # seconds a pan/tilt command stays "live"


@dataclass
class DisplayConfig:
    """Display window size (live PTZ viewer)."""
    win_w: int = 1920
    win_h: int = 1080


@dataclass
class OfflineConfig:
    """Configuration for offline video-processing scripts."""
    video_path:       str   = "/home/guy/experiment_07_05/rgb-01/13_45.mp4"
    output_path:      str   = "/home/guy/experiment_07_05/rgb-01/13_45_motion_only_norfair.avi"
    start_sec:        float = 270.0
    show_window:      bool  = False
    min_motion_area:  float = 2.0
    max_motion_area:  float = float("inf")
    cluster_distance: float = 80.0
    ground_mask_interval_s: float = 120.0
    distance_threshold:  float = 120.0
    hit_counter_max:     int   = 200
    initialization_delay: int  = 2


@dataclass
class PipelineConfig:
    """Top-level config for the live PTZ pipeline.

    Passed to LivePtzPipeline on construction so the pipeline has no
    module-level CONFIG constants of its own.
    """
    camera:  CameraConfig    = field(default_factory=CameraConfig)
    zoom:    ZoomConfig       = field(default_factory=ZoomConfig)
    pid:     PidConfig        = field(default_factory=PidConfig)
    motion:  MotionConfig     = field(default_factory=MotionConfig)
    tracker: TrackerConfig    = field(default_factory=TrackerConfig)
    yolo:    YoloConfig       = field(default_factory=YoloConfig)
    track:   TrackConfig      = field(default_factory=TrackConfig)
    frozen:  FrozenConfig     = field(default_factory=FrozenConfig)
    home:    HomeConfig        = field(default_factory=HomeConfig)
    cmd:     PtzCommandConfig  = field(default_factory=PtzCommandConfig)
    display: DisplayConfig     = field(default_factory=DisplayConfig)
    offline: OfflineConfig     = field(default_factory=OfflineConfig)
