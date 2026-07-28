# -*- coding: utf-8 -*-
"""Detect thread orchestrator for the live PTZ pipeline.

``DetectWorker`` initialises per-thread resources, runs the per-frame loop, and
dispatches to the IDLE/TRACK state handlers in ``idle.py`` / ``track.py``::

    worker = DetectWorker(cfg, ptz, state, classifier)
    threading.Thread(target=worker.run, daemon=True, name="detect").start()

Cross-thread communication is done exclusively through the ``SharedState``
instance passed at construction — the worker never writes to ``LivePtzPipeline``
or ``DisplayLoop`` attributes directly.
"""

__all__ = ["DetectWorker"]

import time
from dataclasses import dataclass, field
from typing import Optional

from dronetracker.config.schema import PipelineConfig
from dronetracker.pipeline.shared_state import SharedState
from dronetracker.pipeline.detect.idle import step_idle
from dronetracker.pipeline.detect.track import step_track
from dronetracker.ptz.pid import PIDController
from dronetracker.ptz.follow import TargetFollower


@dataclass
class _DetectCtx:
    """Mutable per-frame state owned exclusively by the detect thread.

    Kept as a plain dataclass so that all detect-thread state is visible in one
    place and the detect loop is free of instance-mutation races (the only thread
    that touches these fields is the detect thread itself).
    """
    # Ego-motion calibration
    calib_k_pan:     float = 0.0
    calib_k_tilt:    float = 0.0
    calib_n:         int   = 0      # reliable samples; caps at 200
    prev_calib_pose: Optional[tuple] = None   # (pan, tilt) previous frame

    # Zoom blackout
    was_zooming:     bool = False

    # State machine
    state:           str = "IDLE"
    locked_id:       Optional[int] = None
    last_mask_t:     float = field(default_factory=time.time)
    auto_cooldown_until: float = 0.0  # auto-engage suppressed until this wall-clock time

    # TRACK state (lock-on + classify)
    lock_t:          float = 0.0
    focus_done:      bool = False
    focus_t:         float = 0.0
    classify_done:   bool = False
    miss_frames:     int = 0    # consecutive YOLO checks with no detection; resets on any hit
    last_monitor_t:  float = 0.0  # timestamp of last post-classify YOLO check (throttle)
    focus_best:      float = -1.0  # running-max Laplacian variance during AF wait
    focus_stale:     int   = 0     # consecutive frames without a new sharpness max (plateau)
    last_track_cx:   float = 0.0  # last target centre X (follow-off fallback for gate)
    last_track_cy:   float = 0.0  # last target centre Y

    # FPS tracking
    fps_t:   float = field(default_factory=time.time)
    fps_n:   int   = 0
    fps_val: float = 0.0   # smoothed detect-loop fps; used for velocity lead at lock-on
    prev_frame_id: Optional[int] = None   # id(frame) to detect new frames

    # Monotonic count of processed frames; used to key detection/track JSON logs.
    frame_index: int = 0

    def reset_track_fields(self) -> None:
        """Reset the per-lock TRACK-state fields to their fresh-lock defaults.

        Shared by the lock-on entry (idle.py) and the TRACK->IDLE reset so both
        start a (potential) next lock from an identical clean slate.
        """
        self.lock_t         = 0.0
        self.focus_done     = False
        self.focus_t        = 0.0
        self.focus_best     = -1.0
        self.focus_stale    = 0
        self.classify_done  = False
        self.miss_frames    = 0
        self.last_monitor_t = 0.0
        self.last_track_cx  = 0.0
        self.last_track_cy  = 0.0


class DetectWorker:
    """Detect + control thread: initialise resources, dispatch to IDLE/TRACK."""

    def __init__(self, cfg: PipelineConfig, ptz, state: SharedState, classifier):
        self._cfg          = cfg
        self._ptz          = ptz
        self._state        = state
        self._classifier   = classifier
        self._track_filter = None   # lazy-init on first IDLE frame
        self._pid      = PIDController(
            kp       = cfg.follow_pid.kp,
            ki       = cfg.follow_pid.ki,
            kd       = cfg.follow_pid.kd,
            max_out  = cfg.follow_pid.max_out,
            deadband = cfg.follow_pid.deadband,
            i_limit  = cfg.follow_pid.i_limit,
        )
        self._follower = TargetFollower(cfg.frozen)

        # Auto-engage ("auto-hunt") evaluator + optional feature logger.
        from dronetracker.tracking.auto_engage import AutoEngageEvaluator, FeatureLogger
        ae_cfg = cfg.auto_engage
        ae_logger = FeatureLogger(ae_cfg.log_path) if ae_cfg.log_enabled else None
        self._auto = AutoEngageEvaluator(ae_cfg, logger=ae_logger)

        # Per-thread objects — set by run() after lazy imports + first-frame init.
        # Only the detect thread reads/writes these after run() starts.
        self._mt                   = None   # MotionDetector
        self._gmg                  = None   # GroundMaskGenerator
        self._tracker              = None   # Norfair Tracker (replaced on reset)
        self._make_tracker         = None   # () -> fresh Tracker
        self._motion_to_detections = None   # dronetracker.detection.clustering fn

    # ══════════════════════════════════════════════════════════════════════════
    # DETECT THREAD ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Detect + control thread body (call from threading.Thread)."""
        from dronetracker.vision.motion_detector import MotionDetector
        from dronetracker.vision.ground_mask import GroundMaskGenerator
        from norfair import Tracker
        from dronetracker.detection.clustering import motion_to_detections
        from dronetracker.ptz.transform import _zoom_mag

        cfg = self._cfg

        # ── Wait for first frame ──────────────────────────────────────────────
        while not self._state.stop_ev.is_set():
            fr = self._state.get_frame()
            if fr is not None:
                break
            time.sleep(0.05)

        # ── Initialise per-thread detectors ───────────────────────────────────
        self._gmg = GroundMaskGenerator()
        self._gmg.compute(fr)
        self._mt  = MotionDetector(ground_mask_generator=self._gmg)
        self._motion_to_detections = motion_to_detections

        def _make_tracker():
            return Tracker(
                distance_function    = "euclidean",
                distance_threshold   = cfg.tracker.distance_threshold,
                hit_counter_max      = cfg.tracker.hit_counter_max,
                initialization_delay = cfg.tracker.initialization_delay,
            )
        self._make_tracker = _make_tracker
        self._tracker      = _make_tracker()

        ctx = _DetectCtx(
            last_mask_t = time.time(),
            fps_t       = time.time(),
        )

        # Wait for PTZ to finish startup zoom-to-minimum
        for _ in range(100):
            if self._ptz.ready:
                break
            time.sleep(0.1)

        # ── Per-frame loop ────────────────────────────────────────────────────
        while not self._state.stop_ev.is_set():
            fr = self._state.get_frame()
            if fr is None or id(fr) == ctx.prev_frame_id:
                time.sleep(0.008)
                continue
            ctx.prev_frame_id = id(fr)
            ctx.frame_index += 1

            now = time.time()
            do_lock, do_unlock = self._state.drain_commands()

            # ── R key: return home from any state ─────────────────────────────
            if do_unlock:
                ctx = self._reset_to_idle(ctx, reason="operator")
                continue

            # Skip frames during zoom (lens moving → optical flow unreliable)
            if self._ptz._zooming:
                ctx.was_zooming = True
                self._state.publish_det(state=ctx.state)
                continue

            # ── State dispatch ────────────────────────────────────────────────
            if ctx.state == "IDLE":
                fh, fw   = fr.shape[:2]
                mag      = _zoom_mag(self._ptz.zoom_pos)
                pose_now = self._ptz.get_pose()
                pan_now, tilt_now = pose_now[0], pose_now[1]
                skip_calib      = ctx.was_zooming
                ctx.was_zooming = False
                ctx = step_idle(
                    self, fr, fh, fw, now, mag, pan_now, tilt_now,
                    skip_calib, do_lock, ctx,
                )
                self._state.publish_det(state=ctx.state)

            elif ctx.state == "TRACK":
                ctx.was_zooming = False
                if step_track(self, ctx, now, fr):
                    ctx = self._reset_to_idle(ctx, reason="no_drone")

            # FPS update
            ctx.fps_n += 1
            if now - ctx.fps_t >= 1.0:
                ctx.fps_val = ctx.fps_n / (now - ctx.fps_t)
                self._state.publish_det(fps=ctx.fps_val)
                ctx.fps_n = 0
                ctx.fps_t = now

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_to_idle(self, ctx, reason="") -> _DetectCtx:
        """Return the pipeline to IDLE and issue go_home."""
        from dronetracker.tracking.trails import clear_trails
        print(f"[IDLE] {reason} → home")
        ctx.reset_track_fields()
        ctx.state          = "IDLE"
        ctx.locked_id      = None
        ctx.last_mask_t    = time.time()
        ctx.auto_cooldown_until = time.time() + self._cfg.auto_engage.cooldown_s
        self._auto.reset()
        clear_trails()
        if self._track_filter is not None:
            self._track_filter.reset()
        self._pid.reset()
        self._follower.reset()
        self._ptz.go_home()
        self._state.clear_locked_id()
        self._state.clear_yolo_all()
        self._state.publish_det(request_shot=False)
        self._tracker = self._make_tracker()
        return ctx
