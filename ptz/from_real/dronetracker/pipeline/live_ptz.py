"""Live PTZ tracking pipeline orchestrator.

``LivePtzPipeline`` replaces the flat ``grab`` / ``detect`` / ``main`` functions
in the original ``motion_ptz_pipeline.py`` with a class whose methods carry the
same logic but are named and scoped.  The threading model is preserved exactly:

  grab thread  → ``_grab_loop()``        (daemon, reads RTSP, writes latest_frame)
  detect thread → ``_detect_loop()``     (daemon, runs motion+YOLO+control)
  main thread   → ``run()`` display loop (owns the cv2 window + keyboard)

Cross-thread state lives in ``SharedState``; per-detect-thread state lives in a
``_DetectCtx`` dataclass so the detect loop is free of instance mutation races.
"""

__all__ = ["LivePtzPipeline"]

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from dronetracker.config.schema import PipelineConfig
from dronetracker.pipeline.shared_state import SharedState
from dronetracker.media.rtsp import RtspGrabber
from dronetracker.rendering.hud import HudState, draw_hud
from dronetracker.rendering.draw import draw_tracks_overlay, draw_yolo_box
from dronetracker.vision.tiling import split_into_tiles
from dronetracker.vision.nms import global_nms
from dronetracker.tracking.selection import pick_best_track
from dronetracker.media.video import make_writer
from dronetracker.ptz.calibration import update_ego_gain


@dataclass
class _DetectCtx:
    """Mutable per-frame state for the detect thread.

    Kept separate from pipeline attributes so that the only thread
    that touches these fields is the detect thread itself — no locks needed.
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

    # TRACK state (lock-on + classify)
    lock_t:          float = 0.0
    focus_done:      bool = False
    focus_t:         float = 0.0
    classify_done:   bool = False
    miss_frames:     int = 0    # consecutive YOLO checks with no detection; resets on any hit
    last_monitor_t:  float = 0.0  # timestamp of last post-classify YOLO check (throttle)
    focus_best:      float = -1.0  # running-max Laplacian variance during AF wait
    focus_stale:     int   = 0     # consecutive frames without a new sharpness max (plateau)

    # FPS tracking
    fps_t:   float = field(default_factory=time.time)
    fps_n:   int   = 0
    fps_val: float = 0.0   # smoothed detect-loop fps; used for velocity lead at lock-on
    prev_frame_id: Optional[int] = None   # id(frame) to detect new frames


class LivePtzPipeline:
    """Full live PTZ pipeline: grab → detect/control → display."""

    # EMA + calibration constants (not user-tunable; derived from UNV lens)
    _CALIB_EMA    = 0.85
    _CALIB_MIN_DP = 2e-4
    _CALIB_MIN_N  = 3
    _TRACK_MISS_FRAMES = 10   # consecutive YOLO misses before auto-unlock

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg

        # Construct PTZ controller with config values
        from dronetracker.ptz.controller import PTZController
        self._ptz = PTZController(
            cfg.camera.ip, cfg.camera.port, cfg.camera.user, cfg.camera.password,
            zoom_search_pos = cfg.zoom.search_pos,
            zoom_track_pos  = cfg.zoom.track_pos,
            zoom_max        = cfg.zoom.max_pos,
            zoom_step       = cfg.zoom.step,
            zoom_full_s     = cfg.zoom.full_travel_s,
            cmd_ttl         = cfg.cmd.cmd_ttl,
            kp_home         = cfg.home.kp,
            home_tol        = cfg.home.tol,
            home_timeout    = cfg.home.timeout_s,
            home_max_vel    = cfg.home.max_vel,
            min_vel_scale   = cfg.track.min_vel_scale,
        )

        from dronetracker.ptz.pid import PIDController
        self._pid = PIDController(
            cfg.pid.kp, cfg.pid.ki, cfg.pid.kd,
            cfg.pid.max_out, cfg.pid.deadband, cfg.pid.i_limit,
        )

        self._state = SharedState(initial_pid_kp=cfg.pid.kp)
        self._state.pid_gains = [cfg.pid.kp, cfg.pid.ki, cfg.pid.kd]

        # YOLO model (loaded after _patch_tv() in the entry point)
        self._yolo        = None   # set in _init_yolo()
        self._yolo_device = "cpu"
        self._yolo_half   = False
        self._cfg         = cfg

    # ── Startup ────────────────────────────────────────────────────────────────

    def _init_yolo(self) -> None:
        """Load and warm up the YOLO model (call before starting threads).

        Warmup uses a batch of YOLO_WARMUP_TILES zero images so that a TensorRT
        engine's CUDA context is fully built for the runtime batch size before the
        first live classify — otherwise the first inference call bears the full
        JIT-compilation stall.  8 tiles matches a 1920×1080 stream @640 px tiles;
        the extra warmup cost on non-TRT backends is negligible.
        """
        import torch
        from ultralytics import YOLO
        cfg = self._cfg.yolo
        self._yolo_device = 0 if torch.cuda.is_available() else "cpu"
        self._yolo_half   = torch.cuda.is_available()
        self._yolo = YOLO(cfg.model_path, task="detect")

        # Determine backend for the log line
        model_path = cfg.model_path
        backend = "tensorrt" if model_path.endswith(".engine") else "pytorch"

        # Warm up with a full-batch of blank tiles so TRT builds its context now.
        # warmup_tiles must equal the --batch used when building the .engine;
        # see config key yolo.warmup_tiles (default 8 → 1920×1080 @640 px tiles).
        warmup_batch = [np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8)] * cfg.warmup_tiles
        self._yolo(
            warmup_batch,
            imgsz=cfg.imgsz, conf=cfg.conf,
            device=self._yolo_device, half=self._yolo_half,
            verbose=False,
        )
        print(f"[Model] {model_path}  backend={backend}  GPU:{torch.cuda.is_available()}")

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Load YOLO, start grab + detect threads, then run the display loop."""
        self._init_yolo()
        grabber = RtspGrabber(
            self._cfg.camera.rtsp_main,
            self._cfg.camera.rtsp_alt,
            self._state,
        )
        grabber.start()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        time.sleep(3)   # let grab + PTZ startup complete
        self._display_loop()

    # ══════════════════════════════════════════════════════════════════════════
    # GRAB THREAD
    # ══════════════════════════════════════════════════════════════════════════
    # Handled by RtspGrabber — see dronetracker/media/rtsp.py

    # ══════════════════════════════════════════════════════════════════════════
    # DETECT THREAD
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_loop(self) -> None:
        """Detect + control thread body (runs as a daemon thread)."""
        import utils.MotionDetector as _MD
        from utils.MotionDetector import MotionDetector
        from utils.GroundMaskGenerator import GroundMaskGenerator
        from norfair import Tracker
        from dronetracker.detection.clustering import motion_to_detections
        from dronetracker.ptz.transform import PTZTransformation, _zoom_mag, _F_WIDE, _F_TELE

        cfg = self._cfg

        # ── Wait for first frame ──────────────────────────────────────────────
        while not self._state.stop_ev.is_set():
            fr = self._state.get_frame()
            if fr is not None:
                break
            time.sleep(0.05)

        # ── Initialise per-thread detectors ───────────────────────────────────
        gmg = GroundMaskGenerator()
        gmg.compute(fr)
        mt = MotionDetector(ground_mask_generator=gmg)

        def _make_tracker():
            return Tracker(
                distance_function="euclidean",
                distance_threshold=cfg.tracker.distance_threshold,
                hit_counter_max=cfg.tracker.hit_counter_max,
                initialization_delay=cfg.tracker.initialization_delay,
            )
        tracker = _make_tracker()

        ctx = _DetectCtx(
            last_mask_t   = time.time(),
            fps_t         = time.time(),
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

            now    = time.time()

            do_lock, do_unlock = self._state.drain_commands()

            # ── R key: return home from any state ─────────────────────────────
            if do_unlock:
                ctx, tracker = self._reset_to_idle(
                    ctx, mt, _MD, _make_tracker, reason="operator")
                continue   # skip this frame; detect next

            # Skip frames during zoom (lens moving → optical flow unreliable)
            if self._ptz._zooming:
                ctx.was_zooming = True
                self._state.publish_det(state=ctx.state)
                continue

            # ── State dispatch ────────────────────────────────────────────────
            if ctx.state == "IDLE":
                fh, fw = fr.shape[:2]
                mag      = _zoom_mag(self._ptz.zoom_pos)
                pose_now = self._ptz.get_pose()
                pan_now, tilt_now, zoom_now = pose_now[0], pose_now[1], pose_now[2]
                skip_calib   = ctx.was_zooming
                ctx.was_zooming = False
                ctx, tracker = self._step_idle(
                    fr, fh, fw, now, mag, pan_now, tilt_now, zoom_now,
                    skip_calib, do_lock,
                    mt, gmg, tracker, _make_tracker, ctx,
                    motion_to_detections, _MD, _F_WIDE, _F_TELE,
                )
                # Publish state for IDLE (TRACK publishes its own state internally)
                self._state.publish_det(state=ctx.state)
            elif ctx.state == "TRACK":
                ctx.was_zooming = False
                if self._step_track(ctx, now, fr):
                    ctx, tracker = self._reset_to_idle(
                        ctx, mt, _MD, _make_tracker, reason="no_drone")

            # FPS update
            ctx.fps_n += 1
            if now - ctx.fps_t >= 1.0:
                ctx.fps_val = ctx.fps_n / (now - ctx.fps_t)
                self._state.publish_det(fps=ctx.fps_val)
                ctx.fps_n = 0
                ctx.fps_t = now

    # ── OSD mask helper ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_osd_mask(fr, cfg):
        """Return a copy of fr with the OSD timestamp region zeroed out.

        The copy is used only for motion detection — the display frame is
        untouched.  osd_mask_rect = [x, y, w, h]; empty list disables.
        """
        rect = cfg.motion.osd_mask_rect
        if not rect or len(rect) < 4:
            return fr
        import numpy as np
        x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        out = fr.copy()
        out[y:y + h, x:x + w] = 0
        return out

    # ── IDLE step ─────────────────────────────────────────────────────────────

    def _step_idle(self, fr, fh, fw, now, mag, pan_now, tilt_now, zoom_now,
                   skip_calib, do_lock,
                   mt, gmg, tracker, make_tracker, ctx,
                   motion_to_detections, _MD, _F_WIDE, _F_TELE):
        cfg = self._cfg

        if now - ctx.last_mask_t > cfg.motion.ground_mask_interval_s:
            gmg.compute(fr)
            ctx.last_mask_t = now

        det_fr = self._apply_osd_mask(fr, cfg)
        motion_mask = mt.detect_motion(det_fr)
        detections  = motion_to_detections(
            motion_mask,
            min_area=cfg.motion.min_area,
            max_area=cfg.motion.max_area,
            cluster_distance=cfg.motion.cluster_distance,
        )
        tracked = tracker.update(detections=detections)

        # Online calibration from arrow-key moves
        if (not skip_calib and ctx.prev_calib_pose is not None
                and mt.last_affine is not None and mag > 0.0):
            ctx = self._update_calibration(ctx, mt, pan_now, tilt_now, mag)
        ctx.prev_calib_pose = (pan_now, tilt_now)

        self._state.publish_det(
            tracks=list(tracked), locked_id=None, pid_err=(0.0, 0.0),
        )

        if do_lock:
            best = pick_best_track(tracked, fw, fh)
            if best is not None:
                best_obj = next((o for o in tracked if o.id == best), None)
            else:
                best_obj = None

            if best_obj is not None:
                # ── 1. Zoom target (must come first — settle_s depends on it) ──
                zoom_pos_now = self._ptz.zoom_pos   # snapshot before dispatch
                f_now = zoom_pos_now * (_F_TELE - _F_WIDE) + _F_WIDE
                zoom_pos_target = float(np.clip(
                    (cfg.frozen.zoom_factor * f_now - _F_WIDE) / (_F_TELE - _F_WIDE),
                    0.0, cfg.zoom.max_pos))
                zoom_burst_s = max(0.4, abs(zoom_pos_target - zoom_pos_now)
                                   * cfg.zoom.full_travel_s)
                settle_s = max(cfg.frozen.move_settle_s, zoom_burst_s)

                # ── 2. Lead the target: aim where it will be after the move+zoom ──
                # Norfair estimate_velocity is in px/frame; convert to px via fps.
                fps = ctx.fps_val if ctx.fps_val > 1.0 else 25.0   # fallback before first tick
                px, py = float(best_obj.estimate[0][0]), float(best_obj.estimate[0][1])
                try:
                    vx, vy = float(best_obj.estimate_velocity[0][0]), float(best_obj.estimate_velocity[0][1])
                except Exception:
                    vx, vy = 0.0, 0.0   # new/degenerate track — no lead
                lead_frames = settle_s * fps
                px = float(np.clip(px + vx * lead_frames, 0.0, fw))
                py = float(np.clip(py + vy * lead_frames, 0.0, fh))

                # ── 3. Pixel offset → FOV-space PTZ move ──
                ex = px / fw - 0.5
                ey = py / fh - 0.5
                dx = float(np.clip(
                    cfg.frozen.fov_sign_x * cfg.frozen.fov_gain * 2.0 * ex,
                    -1.0, 1.0))
                dy = float(np.clip(
                    cfg.frozen.fov_sign_y * cfg.frozen.fov_gain * 2.0 * ey,
                    -1.0, 1.0))

                from dronetracker.ptz.controller import TRANSLATION_SPACE_FOV
                # Try once; if gated (PTZ busy), retry immediately in case a
                # background flag cleared between iterations.
                accepted = self._ptz.center_and_zoom(
                    dx, dy, zoom_pos_target,
                    space=TRANSLATION_SPACE_FOV,
                    parallel=cfg.frozen.zoom_parallel)
                if not accepted:
                    time.sleep(0.05)   # yield 50 ms so background PTZ flags can clear
                    accepted = self._ptz.center_and_zoom(
                        dx, dy, zoom_pos_target,
                        space=TRANSLATION_SPACE_FOV,
                        parallel=cfg.frozen.zoom_parallel)
                if not accepted:
                    print("[IDLE] center_and_zoom rejected after retry (PTZ busy); staying IDLE")
                    return ctx, tracker
                ctx.locked_id      = best
                ctx.lock_t         = now
                ctx.focus_done     = False
                ctx.focus_t        = 0.0
                ctx.focus_best     = -1.0
                ctx.focus_stale    = 0
                ctx.classify_done  = False
                ctx.miss_frames    = 0
                ctx.last_monitor_t = 0.0
                ctx.state         = "TRACK"
                self._state.publish_det(tracks=[])
                print(f"[TRACK] locked ID={best}  "
                      f"aim=({px:.0f},{py:.0f}) ex={ex:.3f} ey={ey:.3f}  "
                      f"dx={dx:.3f} dy={dy:.3f}  lead={lead_frames:.1f}f v=({vx:.2f},{vy:.2f})  "
                      f"zoom {zoom_pos_now:.3f}→{zoom_pos_target:.3f}")
            else:
                print("[IDLE] no confirmed tracks to lock onto")

        return ctx, tracker

    # ── TRACK step ────────────────────────────────────────────────────────────

    def _step_track(self, ctx: _DetectCtx, now: float, fr) -> bool:
        """Manage autofocus + classify timing.  Returns True if TRACK should exit."""
        cfg = self._cfg
        self._state.publish_det(state="TRACK", locked_id=ctx.locked_id)

        # Phase A — record the moment TRACK starts (first frame after zoom clears).
        # _step_track is only reached after _ptz._zooming clears (detect loop line 238),
        # so the zoom burst has already finished and _do_zoom() has already issued one
        # _autofocus() call.  We do NOT fire a second AF here — that was the source of
        # the double-hunt.  Phase B will watch that single post-zoom AF converge.
        if not ctx.focus_done:
            ctx.focus_done  = True
            ctx.focus_t     = now
            ctx.focus_best  = -1.0
            ctx.focus_stale = 0
            print("[TRACK] AF settling (issued post-zoom)")
            return False

        # Phase B — sharpness-gated classify.
        # Each frame we measure Laplacian variance on a center ROI; declare AF converged
        # when sharpness has plateaued (focus_plateau_frames consecutive non-improving
        # frames).  Classify as soon as:
        #   • pan/tilt has settled  (now - lock_t > move_settle_s), AND
        #   • AF converged          (focus_stale >= focus_plateau_frames), OR
        #   • hard-cap timeout      (now - focus_t > focus_settle_s)
        # Zero extra ONVIF calls — no FPS impact.
        if ctx.focus_done and not ctx.classify_done:
            gray_roi  = self._center_gray_roi(fr, cfg.frozen.focus_roi_frac)
            sharpness = self._roi_sharpness(gray_roi)
            if sharpness > ctx.focus_best:
                ctx.focus_best  = sharpness
                ctx.focus_stale = 0
            else:
                ctx.focus_stale += 1

            pan_tilt_settled = (now - ctx.lock_t)  > cfg.frozen.move_settle_s
            af_converged     = ctx.focus_stale      >= cfg.frozen.focus_plateau_frames
            timed_out        = (now - ctx.focus_t)  > cfg.frozen.focus_settle_s

            if not (pan_tilt_settled and (af_converged or timed_out)):
                return False   # still waiting

            if timed_out and not af_converged:
                print(f"[TRACK] AF timeout after {now - ctx.focus_t:.2f}s — classify anyway")
            else:
                print(f"[TRACK] AF converged in {now - ctx.focus_t:.2f}s  "
                      f"sharpness={ctx.focus_best:.1f}  stale={ctx.focus_stale}")

            # One-shot classify pass
            dets = self._run_yolo_tiled(fr)
            if dets:
                best = max(dets, key=lambda d: d[4])
                self._state.publish_det(yolo_box=best)
            else:
                self._state.clear_yolo_box()
            self._state.set_yolo_boxes(dets)
            self._save_classified_crops(fr, dets)
            self._state.publish_det(request_shot=True)
            ctx.classify_done  = True
            ctx.last_monitor_t = now   # classify counts as first check; next YOLO waits one interval
            ctx.miss_frames = 0 if dets else 1   # start counting if classify itself found nothing
            print(f"[TRACK] classify done — {len(dets)} detection(s)")
            return False

        if ctx.classify_done:
            # Throttled monitoring: run YOLO at most every monitor_interval_s.
            # Skipped frames return immediately → loop runs at frame rate (~30 fps).
            # The display overlay keeps the last published boxes between checks.
            if (now - ctx.last_monitor_t) < cfg.frozen.monitor_interval_s:
                return False
            ctx.last_monitor_t = now
            dets = self._run_yolo_tiled(fr)
            if dets:
                best = max(dets, key=lambda d: d[4])
                self._state.publish_det(yolo_box=best)
                self._state.set_yolo_boxes(dets)
                ctx.miss_frames = 0
            else:
                self._state.clear_yolo_all()
                ctx.miss_frames += 1
                if ctx.miss_frames >= self._TRACK_MISS_FRAMES:
                    print(f"[TRACK] no detection for {self._TRACK_MISS_FRAMES} checks → IDLE")
                    return True   # caller resets

        return False

    # ── Focus-sharpness helpers (pure CPU, no ONVIF — safe on detect thread) ──

    @staticmethod
    def _center_gray_roi(frame: np.ndarray, roi_frac: float) -> np.ndarray:
        """Return a grayscale center crop — (roi_frac × min_dim) square.

        Args:
            frame:    BGR or grayscale frame from the grab thread.
            roi_frac: Fraction of min(height, width) to use as the ROI side.
        """
        import cv2
        fh, fw = frame.shape[:2]
        half = max(1, int(min(fh, fw) * roi_frac / 2))
        cy, cx = fh // 2, fw // 2
        roi = frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        if frame.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi

    @staticmethod
    def _roi_sharpness(gray_roi: np.ndarray) -> float:
        """Laplacian variance of a grayscale ROI — cheap focus-sharpness metric.

        Higher value = sharper image.  Sub-millisecond even for large ROIs.

        Args:
            gray_roi: 2-D uint8 grayscale array (from _center_gray_roi or similar).
        """
        import cv2
        if gray_roi.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

    # ── YOLO helpers ──────────────────────────────────────────────────────────

    def _run_yolo_tiled(self, frame):
        """Run tiled YOLO: split into cfg.yolo.imgsz tiles, infer all tiles in one batched pass, merge via NMS.

        Padding ensures every tile is exactly (sz×sz) so Ultralytics stacks them into a
        single tensor and executes one GPU forward pass for the entire frame.

        Returns detections as [(x1, y1, x2, y2, conf, label), ...] in full-frame pixel coords.
        """
        t0  = time.perf_counter()
        cfg = self._cfg.yolo
        sz  = cfg.imgsz
        tiles, coords = split_into_tiles(frame, (sz, sz))

        # Pad pass — build a uniform-shape batch; keeps coords index-aligned with results.
        batch = []
        for tile in tiles:
            # Pad edge tiles that are smaller than sz so YOLO never letterboxes
            # internally (which would shift the returned xyxy coords).
            if tile.shape[0] < sz or tile.shape[1] < sz:
                padded = np.zeros((sz, sz, 3), dtype=tile.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            batch.append(tile)

        if not batch:
            print(f"[YOLO] total=  0.0ms  infer=  0.0ms  tiles=0  dets=0  loop_fps={self._state.fps:.1f}, device={self._yolo_device}")
            return []

        # Single batched inference call — one GPU forward pass for all tiles.
        t_inf0 = time.perf_counter()
        results = self._yolo(
            batch,
            imgsz=sz, conf=cfg.conf,
            device=self._yolo_device, half=self._yolo_half,
            verbose=False,
        )
        t_inf1 = time.perf_counter()

        # Stitch pass — apply per-tile pixel offsets and collect all detections.
        all_boxes, all_scores, all_classes = [], [], []
        for res, (ox, oy) in zip(results, coords):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for box in res.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                all_boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
                all_scores.append(float(box.conf[0]))
                all_classes.append(int(box.cls[0]))

        dt_total = (time.perf_counter() - t0) * 1e3
        dt_infer = (t_inf1 - t_inf0) * 1e3

        if not all_boxes:
            print(f"[YOLO] total={dt_total:6.1f}ms  infer={dt_infer:6.1f}ms  "
                  f"tiles={len(batch)}  dets=0  loop_fps={self._state.fps:.1f}")
            return []
        kept_boxes, kept_scores, kept_classes = global_nms(all_boxes, all_scores, all_classes, 0.4)
        names = self._yolo.names or {}
        out = [
            (int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s), names.get(c, "drone"))
            for b, s, c in zip(kept_boxes, kept_scores, kept_classes)
        ]
        print(f"[YOLO] classify={dt_total:6.1f}ms  infer={dt_infer:6.1f}ms  "
              f"tiles={len(batch)}  dets={len(out)}  loop_fps={self._state.fps:.1f}")
        return out

    def _save_classified_crops(self, frame, dets):
        """Save per-detection bbox crops + one annotated full frame.

        Writes:
          crop_<label>_<conf>_<ts>_<i>.png  — raw bbox region from the frame
          annotated_<ts>.png                — full frame with all boxes drawn
        If dets is empty, saves full raw frame as capture_<ts>.png instead.
        Returns the number of crops written.
        """
        import cv2
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not dets:
            fn = f"capture_{ts}.png"
            cv2.imwrite(fn, frame)
            print(f"[TRACK] no detections — saved full frame {fn}")
            return 0
        count = 0
        for i, (x1, y1, x2, y2, conf, label) in enumerate(dets):
            if y2 <= y1 or x2 <= x1:
                continue
            crop = frame[y1:y2, x1:x2]
            safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
            fn = f"crop_{safe_label}_{conf:.2f}_{ts}_{i}.png"
            cv2.imwrite(fn, crop)
            print(f"[TRACK] saved {fn}")
            count += 1
        annotated = frame.copy()
        for x1, y1, x2, y2, conf, label in dets:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        fn = f"annotated_{ts}.png"
        cv2.imwrite(fn, annotated)
        print(f"[TRACK] saved annotated {fn}")
        return count

    # ── Reset helper ──────────────────────────────────────────────────────────

    def _reset_to_idle(self, ctx, mt, _MD, make_tracker, reason=""):
        """Return the pipeline to IDLE and issue go_home."""
        from dronetracker.tracking.trails import clear_trails
        print(f"[IDLE] {reason} → home")
        ctx.state          = "IDLE"
        ctx.locked_id      = None
        ctx.lock_t         = 0.0
        ctx.focus_done     = False
        ctx.focus_t        = 0.0
        ctx.classify_done  = False
        ctx.miss_frames    = 0
        ctx.last_monitor_t = 0.0
        ctx.focus_best     = -1.0
        ctx.focus_stale    = 0
        ctx.last_mask_t   = time.time()
        self._pid.reset()
        clear_trails()   # wipe trails so recycled Norfair IDs don't inherit old points
        self._ptz.go_home()
        # publish_det(locked_id=None) and (yolo_box=None) are no-ops because
        # the guard uses "is not None".  Use the dedicated clear helpers instead.
        self._state.clear_locked_id()
        self._state.clear_yolo_all()   # atomically clears yolo_box + yolo_boxes
        self._state.publish_det(request_shot=False)
        tracker = make_tracker()
        return ctx, tracker

    # ── Calibration update ─────────────────────────────────────────────────────

    def _update_calibration(self, ctx: _DetectCtx, mt, pan_now, tilt_now, mag):
        """EMA-update ego-motion calibration from the latest affine translation."""
        dpan  = pan_now  - ctx.prev_calib_pose[0]
        dtilt = tilt_now - ctx.prev_calib_pose[1]
        tx = float(mt.last_affine[0, 2])
        ty = float(mt.last_affine[1, 2])

        new_k_pan = update_ego_gain(
            ctx.calib_k_pan, tx, dpan, mag, self._CALIB_EMA, self._CALIB_MIN_DP)
        new_k_tilt = update_ego_gain(
            ctx.calib_k_tilt, ty, dtilt, mag, self._CALIB_EMA, self._CALIB_MIN_DP)

        if new_k_pan != ctx.calib_k_pan or new_k_tilt != ctx.calib_k_tilt:
            ctx.calib_n = min(ctx.calib_n + 1, 200)
            if ctx.calib_n == self._CALIB_MIN_N:
                print(f"[Calib] ego-motion trusted: "
                      f"k_pan={new_k_pan:.1f}  k_tilt={new_k_tilt:.1f}")
        ctx.calib_k_pan  = new_k_pan
        ctx.calib_k_tilt = new_k_tilt
        return ctx

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY THREAD (main thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _display_loop(self) -> None:
        """Main thread: display, HUD, keyboard handling."""
        import cv2
        from dronetracker.ptz.transform import _F_WIDE, _F_TELE
        from dronetracker.tracking.trails import track_history, prune_stale

        cfg = self._cfg
        WIN = "UAV Motion Tracker — UNV IPC6852ER-X45-VF"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, cfg.display.win_w, cfg.display.win_h)

        dfps = 0.0; d_t = time.time(); d_n = 0
        cam_w = cam_h = 0

        while True:
            raw = self._state.get_frame()

            if raw is None:
                disp = np.zeros(
                    (cfg.display.win_h, cfg.display.win_w, 3), np.uint8)
                cv2.putText(disp, "Connecting...",
                            (cfg.display.win_w // 2 - 160, cfg.display.win_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
                snap = {"state": "IDLE", "tracks": [], "locked_id": None,
                        "yolo_box": None, "yolo_boxes": [], "fps": 0.0,
                        "pid_err": (0.0, 0.0), "request_shot": False}
            else:
                cam_h, cam_w = raw.shape[:2]
                disp = cv2.resize(
                    raw.copy(), (cfg.display.win_w, cfg.display.win_h))
                sx = cfg.display.win_w / cam_w
                sy = cfg.display.win_h / cam_h
                snap = self._state.snapshot_for_display()

                now_draw = time.time()
                prune_stale(now_draw, cfg.tracker.trail_max_age_s)
                draw_tracks_overlay(
                    disp, snap["tracks"], snap["locked_id"], sx, sy,
                    track_history, now=now_draw,
                )
                for box in snap["yolo_boxes"]:
                    draw_yolo_box(disp, box, sx, sy)

            d_n += 1
            now_d = time.time()
            if now_d - d_t >= 1.0:
                dfps = d_n / (now_d - d_t); d_n = 0; d_t = now_d

            with self._state.det_lock:
                cur_state = self._state.state

            # Build HUD snapshot (no globals)
            f_mm  = self._ptz.zoom_pos * (_F_TELE - _F_WIDE) + _F_WIDE
            fov   = 2 * math.degrees(math.atan(7.18 / (2 * f_mm)))
            hud   = HudState(
                state       = snap["state"],
                locked_id   = snap["locked_id"],
                dfps        = dfps,
                det_fps     = snap["fps"],
                zoom_pos    = self._ptz.zoom_pos,
                focal_mm    = f_mm,
                fov_deg     = fov,
                pid_err     = snap["pid_err"],
                pid_kp      = self._pid.kp,
                yolo_active = snap["yolo_box"] is not None,
                cam_w       = cam_w,
                cam_h       = cam_h,
                recording   = self._state.recording,
                ptz_ready   = self._ptz.ready,
                ptz_error   = self._ptz.connect_error,
            )
            draw_hud(disp, hud)

            if self._state.recording and self._state.video_writer:
                self._state.video_writer.write(disp)

            cv2.imshow(WIN, disp)

            if self._state.take_shot_request():
                fn = datetime.now().strftime("freeze_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, disp)
                print(f"[TRACK] saved {fn}")

            key_raw = cv2.waitKey(1)
            key     = key_raw & 0xFF
            if key == ord("s"):
                self._handle_manual_screenshot(disp)
            elif self._handle_key(key, key_raw, cur_state):
                break   # Q pressed → exit

        self._state.stop_ev.set()
        self._ptz.stop_all()
        if self._state.recording and self._state.video_writer:
            self._state.video_writer.release()
        cv2.destroyAllWindows()
        print("[Main] stopped.")

    def _handle_key(self, key: int, key_raw: int, cur_state: str) -> bool:
        """Process a single keypress; return True if Q was pressed (quit)."""
        cfg = self._cfg

        if key == ord("q"):
            return True
        elif key == ord("v"):
            self._toggle_recording()
        elif key == ord("z"):
            self._ptz.zoom_in()
            print(f"[Zoom→] {self._ptz.zoom_pos:.2f}")
        elif key == ord("x"):
            self._ptz.zoom_out()
            print(f"[Zoom→] {self._ptz.zoom_pos:.2f}")
        elif key == ord("r"):
            self._state.post_unlock_command()
        elif key in (ord("t"), 13) and cur_state == "IDLE":
            self._state.post_lock_command()
        elif key == ord("p"):
            self._pid.kp = round(min(15.0, self._pid.kp + 0.1), 1)
            self._state.pid_gains[0] = self._pid.kp
            print(f"[PID] Kp={self._pid.kp:.1f}")
        elif key == ord("o"):
            self._pid.kp = round(max(0.1, self._pid.kp - 0.1), 1)
            self._state.pid_gains[0] = self._pid.kp
            print(f"[PID] Kp={self._pid.kp:.1f}")
        elif key == ord("h"):
            self._ptz.save_home()
        elif key_raw == 82 and cur_state == "IDLE":   # Up → tilt up
            self._ptz.move(0.0, cfg.zoom.manual_speed * self._ptz.vel_scale())
        elif key_raw == 84 and cur_state == "IDLE":   # Down → tilt down
            self._ptz.move(0.0, -cfg.zoom.manual_speed * self._ptz.vel_scale())
        elif key_raw == 81 and cur_state == "IDLE":   # Left → pan left
            self._ptz.move(-cfg.zoom.manual_speed * self._ptz.vel_scale(), 0.0)
        elif key_raw == 83 and cur_state == "IDLE":   # Right → pan right
            self._ptz.move(cfg.zoom.manual_speed * self._ptz.vel_scale(), 0.0)
        elif key_raw in (81, 82, 83, 84):
            pass   # arrow in TRACK — silently ignored
        elif key_raw != -1:
            print(f"key = {key_raw}")

        return False

    def _handle_manual_screenshot(self, disp) -> None:
        """Save a manual screenshot to disk."""
        import cv2
        fn = datetime.now().strftime("shot_%Y%m%d_%H%M%S.png")
        cv2.imwrite(fn, disp)
        print(f"[Shot] {fn}")

    def _toggle_recording(self) -> None:
        """Start or stop AVI recording to a timestamped file."""
        cfg = self._cfg
        if not self._state.recording:
            fn = datetime.now().strftime("rec_%Y%m%d_%H%M%S.avi")
            self._state.video_writer = make_writer(
                fn, 25.0, cfg.display.win_w, cfg.display.win_h)
            self._state.recording = True
            print(f"[Rec] → {fn}")
        else:
            self._state.recording = False
            if self._state.video_writer:
                self._state.video_writer.release()
                self._state.video_writer = None
            print("[Rec] stopped")
