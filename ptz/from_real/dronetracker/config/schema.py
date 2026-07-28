"""Typed configuration dataclasses for the DroneTracker system.

Every field has a default that matches the hard-coded constants in the live
PTZ pipeline so the system works out-of-the-box without a config file.
Override via config.yaml / .env (see loader.py).
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
    # Full RTSP URL overrides.  Empty string => use the computed UNV URL below.
    # Used by the simulator entry point, whose RTSP path differs from the camera's.
    rtsp_main_override: str = ""
    rtsp_alt_override:  str = ""

    @property
    def rtsp_main(self) -> str:
        if self.rtsp_main_override:
            return self.rtsp_main_override
        return (
            f"rtsp://{self.user}:{self.password}@{self.ip}:554"
            "/unicast/c1/s0/live"
        )

    @property
    def rtsp_alt(self) -> str:
        if self.rtsp_alt_override:
            return self.rtsp_alt_override
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
    distance_threshold:  float = 200.0
    hit_counter_max:     int   = 200    # survive full camera-move+settle blackout (~6 s at 30 fps)
    initialization_delay: int  = 2
    trail_max_age_s:     float = 2.0    # seconds before an unseen track's trail is deleted

    # ── Collision-robust data association ────────────────────────────────────
    # Replace the bare "euclidean" distance function with one that adds a
    # *capped* area-mismatch penalty so crossing objects are less likely to
    # swap IDs.  Set False to revert to legacy behaviour (e.g. to regenerate
    # golden-test snapshots against a known baseline).
    use_motion_size_distance: bool  = True
    size_weight:              float = 40.0   # px added per unit of area ratio beyond 1×
    size_penalty_cap:         float = 60.0   # hard cap on size penalty (px)

    # Norfair ReID: keeps dead tracks as candidates for re-identification so a
    # track briefly starved of detections reclaims its old ID rather than
    # spawning a new one.  Only applied when use_motion_size_distance is True.
    reid_enabled:             bool  = True
    reid_distance_threshold:  float = 150.0  # max combined distance to accept a ReID match
    reid_hit_counter_max:     int   = 900    # frames a dead track lingers as a ReID candidate (~30 s)

    # ── Appearance-based ID protection (bounded tie-breaker) ─────────────────
    # Adds a *bounded, dead-banded, size-gated* appearance penalty so a
    # clearly-different object (e.g. an insect crossing a near-stationary drone)
    # is out-competed for a track's ID — without ever rejecting the object's own
    # detection (which would fragment the track).  Only applied when
    # use_motion_size_distance is True.  Set use_appearance False to revert.
    use_appearance:         bool  = True
    appearance_weight:      float = 300.0  # px per unit Bhattacharyya distance (slope)
    appearance_deadband:    float = 0.15   # self-distance below this → 0 penalty (absorbs noise)
    appearance_penalty_cap: float = 50.0   # max appearance penalty (px) — never rejects own match
    appearance_ema_alpha:   float = 0.15   # per-track template EMA blend rate
    appearance_min_box_px:  int   = 3      # skip descriptor for boxes smaller than this
    appearance_min_pixels:  int   = 8      # min motion pixels for a reliable descriptor

    # Stopped-track protection: a mature track (matched on >= maturity_frames)
    # that has then had NO motion detection for >= dormancy_frames — i.e. the
    # object genuinely stopped — switches appearance to a STRONG gate (cap →
    # dormant_cap) so an intruder passing through cannot steal its ID and the
    # real object reclaims it by appearance on resume.  While ACTIVELY tracked
    # (fewer than dormancy_frames missed) appearance stays a bounded tie-breaker
    # (appearance_penalty_cap above) so the object's own detection is never
    # rejected — even if its appearance momentarily changes (blur/defocus).
    appearance_maturity_frames: int   = 45    # matched frames to count as "mature" (~1.5–2 s)
    appearance_dormancy_frames: int   = 10    # no-detection frames before the strong gate engages
    appearance_dormant_cap:     float = 1e9   # strong (uncapped) gate once stopped & mature
    # Slow-object appearance-in-distance gate: superseded by the template tracker
    # below for slow objects, so disabled by default (0).  Set > 0 to re-enable.
    appearance_slow_speed:      float = 0.0

    # ── Template-at-location tracking (near-stationary objects) ──────────────
    # For a mature track moving < template_max_speed px/frame the motion detector
    # usually yields nothing, so we template-match the object's stored appearance
    # patch around its predicted location: if the blob is still there we pin the
    # track to the matched spot (re-localising it) and suppress co-located motion,
    # so a passing object can't steal the ID; if the blob is gone the track is
    # released.  Decoupled from motion association.  Disable with the flag.
    template_track_enabled:   bool  = True
    template_max_speed:       float = 1.0    # px/frame; only for slower tracks
    template_search_radius:   int   = 20     # px search window around the estimate
    template_match_threshold: float = 0.5    # min TM_CCOEFF_NORMED peak = "blob present"
    template_pad:             int   = 4      # px padding around the box → patch
    template_suppress_radius: int   = 10     # px: drop motion dets this close to a synthetic
    template_match_dist:      float = 0.6    # max Bhattacharyya dist for nearby motion to count
                                             # as "the object resuming" (vs a different intruder)


@dataclass
class YoloConfig:
    """YOLO classification and auto-zoom parameters."""
    model_path:     str   = "results/runs/detect/train-15/weights/best.pt"
    imgsz:          int   = 640
    conf:           float = 0.18
    # Must equal the --batch used when exporting the TensorRT engine (if using .engine).
    # 1920×1080 @640 px tiles → 8 (default); 2560×1440 → 15.
    warmup_tiles:   int   = 8


@dataclass
class TrackConfig:
    """TRACK-state control parameters."""
    min_vel_scale: float = 0.50  # floor on pan/tilt velocity at max zoom (follow + arrow keys); higher = faster when zoomed in


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
    # Velocity-lead cap for lock-on aim: the aim leads the target by its velocity
    # over the settle time, but a long zoom-burst settle (~3.5s) × a moving target
    # × a noisy Kalman velocity can fling the aim far off (clipped to a frame edge)
    # so the camera zooms onto empty space.  Cap the lead displacement to this
    # fraction of the frame so the target stays inside the (post-zoom) FOV.
    lockon_max_lead_frac: float = 0.06   # max lead as a fraction of frame W/H (~115px @1920)
    # Zoom-in on lock-on: doubles optical magnification by default.
    zoom_factor:   float = 4.0   # magnification multiplier applied at T/Enter lock
    zoom_parallel: bool  = True  # False → zoom starts after pan/tilt settles (sequential)
    # Sharpness-gated autofocus convergence detection (see _step_track Phase B).
    focus_plateau_frames: int   = 4     # consecutive non-improving Laplacian frames → AF converged
    focus_roi_frac:       float = 0.30  # center-crop fraction of min(h,w) for sharpness measurement
    # Post-classify monitoring throttle: run full-frame tiled YOLO at most this often.
    # Keeps detect-loop FPS high while still checking for sustained drone loss.
    monitor_interval_s:   float = 0.1   # min seconds between post-classify YOLO passes (~10×/sec)
    # Consecutive YOLO misses (in the monitor phase) before auto-unlock TRACK → IDLE.
    track_miss_frames:    int   = 10

    # ── Continuous follow loop ─────────────────────────────────────────────────
    # Set follow_enabled: true in config.yaml to activate YOLO-driven PTZ follow.
    # All other fields take effect only when follow_enabled is true.
    follow_enabled:          bool  = False  # master switch; False = today's sit-still behaviour
    follow_lead_s:           float = 0.1    # actuation lead: predict this far ahead (seconds)
    follow_vel_ema:          float = 0.5    # EMA alpha blending instantaneous into running velocity
    follow_gate_px:          float = 150.0  # max px from prediction to associate a YOLO box
    follow_coast_frames:     int   = 3      # coast (decaying) misses before holding still
    follow_coast_decay:      float = 0.8    # per-coast velocity decay factor
    follow_apply_vel_scale:  bool  = True   # multiply PID output by ptz.vel_scale()
    follow_sign_x:           int   = +1     # flip to -1 if follow drives the wrong pan direction
    follow_sign_y:           int   = +1     # flip to -1 if follow drives the wrong tilt direction


@dataclass
class FollowPidConfig:
    """PID gains for the continuous YOLO-driven PTZ follow loop.

    Kept separate from PidConfig so follow tuning does not entangle the
    legacy manual-joystick PID semantics.
    """
    kp:       float = 2.0    # follow aggressiveness; raise for faster pan/tilt tracking
    ki:       float = 0.0
    kd:       float = 0.05
    max_out:  float = 1.0
    deadband: float = 0.03
    i_limit:  float = 0.2


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
class StorageConfig:
    """SSD output location + rolling size cap for the live PTZ pipeline.

    When ``enabled`` is True, the live pipeline writes each recording chunk into
    its own timestamped folder under ``base_dir`` (``<base_dir>/YYYYmmdd_HHMMSS/``
    holding ``video.avi`` + ``detections.json`` + ``tracks.json`` + ``crops/``).
    A new chunk starts every ``segment_seconds``; once the combined size of all
    chunks exceeds ``max_size_gb`` the oldest *whole* chunk folders are deleted
    so related files (video + its JSON + crops) always come and go together.

    Default ``enabled = False`` keeps the legacy behaviour (recording toggled
    with the 'v' key, written to the current working directory).
    """
    enabled:         bool  = False        # master switch; False = legacy CWD behaviour
    base_dir:        str   = "output"     # SSD mount path; all chunks saved under here
    max_size_gb:     float = 10.0         # combined rolling quota; 0 = unlimited
    segment_seconds: float = 300.0        # rotate to a new chunk every N seconds
    auto_start:      bool  = True         # begin recording automatically on launch
    save_detections: bool  = True         # write detections.json per chunk
    save_tracks:     bool  = True         # write tracks.json per chunk
    save_crops:      bool  = True         # write classified-crop PNGs per chunk


@dataclass
class TrackFilterConfig:
    """Whole-life track filter for offline pipelines (two-pass).

    Tracks that fail any criterion are silently dropped from the annotated
    output video.  Set ``enabled = False`` to bypass the filter entirely
    (falls back to single-pass, draw-everything behaviour).
    """
    enabled:              bool  = True
    min_lifespan_frames:  int   = 15     # drop tracks shorter than this
    jump_window_frames:   int   = 10     # look-back window for jump detection
    jump_frac:            float = 0.2   # max displacement as fraction of frame diagonal
    max_median_step_px:   float = 10.0   # max median consecutive-frame step (px)
    max_step_px:          float = 150.0  # drop track if ANY consecutive-frame step
                                         # exceeds this (px).  0 disables.
    # Drift/sky rejection: drop a track whose detections look like background
    # rather than a real blob — median blob-vs-local-sky intensity contrast below
    # this (measured drone ~40-80, sky-drift ~0-2).  0 disables the criterion.
    min_blob_contrast:    float = 0.0
    min_contrast_samples: int   = 8      # real-detection contrast samples before judging
    # Kill a track that has stayed within stationary_move_px for this many seconds
    # (a static artifact that never moves).  0 disables.
    stationary_kill_s:    float = 30.0
    stationary_move_px:   float = 25.0   # within this radius counts as "not moving"
    # Linear-coast-gap criterion: drop a track that is near-straight AND has a
    # >= max_coast_gap_frames consecutive missed-detection gap within the recent
    # window (Norfair coasting a phantom straight line across undetected frames).
    # Windowed: the track is allowed back once the gap slides out of the window.
    # Set max_coast_gap_frames <= 0 to disable this criterion.
    max_coast_gap_frames:     int   = 5     # consecutive missed-detection frames that mark a coast gap
    coast_window_frames:      int   = 15    # recent look-back (frames) over which a coast gap still counts
    coast_linear_straightness: float = 0.95  # net/path straightness >= this counts as "hard linear"
    coast_min_speed_px:       float = 1.0   # avg speed (px/frame) below this is "near zero" → not linear


@dataclass
class AutoEngageConfig:
    """Automatic IDLE -> TRACK ("auto-hunt") parameters.

    A motion-only, appearance-free drone-likeness gate evaluated on the
    *filtered* (post-TrackFilter) tracks.  Birds fly irregularly (fluctuating
    velocity / heading / curvature + oscillating apparent size); drones are
    smooth and rigid.  All positional features are scale-normalized by the
    blob's apparent size so the gate is zoom-invariant.

    ``enabled`` gates only the *live lock-on action*.  Scoring, on-screen
    annotation, and feature logging always run when an evaluator is present
    (live + offline) so the score can be inspected and a training set built.
    See the two grounding papers: Nesteruk et al. (Sensors 2026) and
    Yao et al. (Drones 2026).
    """
    enabled:             bool  = False   # live auto-engage master switch (safety: off)
    dwell_s:             float = 0.4     # must stay qualified this long before locking
    release_frames:      int   = 15      # once engaged, stay latched on the target through
                                         # score dips (e.g. while it banks/turns); release only
                                         # after it is undetected for this many frames
    cooldown_s:          float = 1.0     # suppress auto-engage this long after any unlock
    history_s:           float = 1.0     # sliding window for fluctuation stats
    history_max:         int   = 120     # hard cap on samples kept per track
    min_detections:      int   = 20      # real (non-coasting) samples before a track can engage
    require_sky:         bool  = True    # only engage targets in the sky region (ground mask == 0)
    min_net_disp_frac:   float = 1.0     # net displacement / apparent-size diagonal (rejects static)
    dronelike_threshold: float = 0.5     # score >= this is "drone-like" (raise to be stricter)
    # Per-feature fluctuation ceilings: a feature at/above its ceiling reads as
    # fully bird-like (contributes 0 to the score); 0 reads as fully drone-like.
    # Defaults are permissive ("never miss a real drone"); tighten via config.
    vel_cv_max:          float = 0.7     # coefficient of variation of step speed
    accel_mean_max:      float = 0.2     # mean |Δspeed| (normalized)
    heading_change_max:  float = 0.6     # proportion of steps that change heading
    curvature_cv_max:    float = 1.0     # coefficient of variation of turn angle
    curvature_mean_max:  float = 1.2     # mean turn angle (radians)
    area_cv_max:         float = 0.6     # coefficient of variation of apparent area
    area_change_max:     float = 0.4     # mean |ΔA|/A between frames
    # Feature-vector logging (builds the dataset to later train a learned scorer).
    log_enabled:         bool  = True
    log_path:            str   = "auto_engage_features.csv"


@dataclass
class SimConfig:
    """PTZ simulator transport parameters (sim_live_ptz entry point only).

    The simulator runs on a separate host; the Jetson-side PTZSimController
    sends ONVIF-equivalent commands to ``host_ip:host_port`` over UDP and
    listens for pose updates on ``listen_port``.  Video frames come from the
    simulator's RTSP stream — point camera.rtsp_main_override at it.
    """
    host_ip:     str = "127.0.0.1"   # simulator host that receives PTZ commands
    host_port:   int = 5005          # UDP port on the simulator for commands
    listen_port: int = 5006          # local UDP port for incoming pose updates
    # Scene to play in the simulator (0 = none). Sent to ptz_sim over the command
    # channel; ptz_sim plays it a fixed delay after Isaac Sim finishes loading.
    scene:       int = 0


@dataclass
class OfflineConfig:
    """Configuration for offline video-processing scripts."""
    video_path:       str   = "/home/guy/experiment_07_05/rgb-01/13_45.mp4"
    output_path:      str   = "/home/guy/experiment_07_05/rgb-01/13_45_motion_only_norfair.avi"
    start_sec:        float = 270.0
    end_sec:          float = -1.0    # -1 means run to end of file
    show_window:      bool  = False
    min_motion_area:  float = 2.0
    max_motion_area:  float = float("inf")
    cluster_distance: float = 80.0
    ground_mask_interval_s: float = 120.0
    # Raised 120 → 200 (matches the live TrackerConfig gate) to leave headroom
    # for the capped size + appearance tie-breakers: position + size_cap(60) +
    # appearance_cap(50) must stay below this so a true match is never rejected.
    distance_threshold:  float = 200.0
    # Coast time after detections stop: a saturated track loses 1/frame, so it
    # survives ~hit_counter_max frames unfed.  Lowered 200 → 30 (~1.2 s at 25 fps)
    # so a fast object that loses its detections dies quickly instead of coasting
    # across the frame.  ReID (reid_hit_counter_max) still allows re-acquisition.
    hit_counter_max:     int   = 30
    initialization_delay: int  = 2

    # ── Collision-robust data association (mirrors TrackerConfig) ────────────
    use_motion_size_distance: bool  = True
    size_weight:              float = 40.0
    size_penalty_cap:         float = 60.0
    reid_enabled:             bool  = True
    reid_distance_threshold:  float = 150.0
    reid_hit_counter_max:     int   = 900

    # ── Appearance-based ID protection (mirrors TrackerConfig) ───────────────
    use_appearance:         bool  = True
    appearance_weight:      float = 300.0
    appearance_deadband:    float = 0.15
    appearance_penalty_cap: float = 50.0
    appearance_ema_alpha:   float = 0.15
    appearance_min_box_px:  int   = 3
    appearance_min_pixels:  int   = 8
    appearance_maturity_frames: int   = 45
    appearance_dormancy_frames: int   = 10
    appearance_dormant_cap:     float = 1e9
    appearance_slow_speed:      float = 0.0

    # Template-at-location tracking (mirrors TrackerConfig)
    template_track_enabled:   bool  = True
    template_max_speed:       float = 1.0
    template_search_radius:   int   = 20
    template_match_threshold: float = 0.5
    template_pad:             int   = 4
    template_suppress_radius: int   = 10
    template_match_dist:      float = 0.6


@dataclass
class PipelineConfig:
    """Top-level config for the live PTZ pipeline.

    Passed to LivePtzPipeline on construction so the pipeline has no
    module-level CONFIG constants of its own.
    """
    camera:       CameraConfig       = field(default_factory=CameraConfig)
    zoom:         ZoomConfig          = field(default_factory=ZoomConfig)
    pid:          PidConfig           = field(default_factory=PidConfig)
    follow_pid:   FollowPidConfig     = field(default_factory=FollowPidConfig)
    motion:       MotionConfig        = field(default_factory=MotionConfig)
    tracker:      TrackerConfig       = field(default_factory=TrackerConfig)
    yolo:         YoloConfig          = field(default_factory=YoloConfig)
    track:        TrackConfig         = field(default_factory=TrackConfig)
    frozen:       FrozenConfig        = field(default_factory=FrozenConfig)
    home:         HomeConfig          = field(default_factory=HomeConfig)
    cmd:          PtzCommandConfig    = field(default_factory=PtzCommandConfig)
    display:      DisplayConfig       = field(default_factory=DisplayConfig)
    offline:      OfflineConfig       = field(default_factory=OfflineConfig)
    track_filter: TrackFilterConfig   = field(default_factory=TrackFilterConfig)
    auto_engage:  AutoEngageConfig     = field(default_factory=AutoEngageConfig)
    storage:      StorageConfig        = field(default_factory=StorageConfig)
    sim:          SimConfig            = field(default_factory=SimConfig)
