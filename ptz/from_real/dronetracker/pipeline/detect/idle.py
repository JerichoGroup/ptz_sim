# -*- coding: utf-8 -*-
"""IDLE state step for the live PTZ detect thread.

``step_idle`` runs once per frame while the state machine is in IDLE: refresh
the ground mask, run motion detection + clustering + Norfair tracking, apply
the optional track filter and ego-motion calibration, then — if a lock command
arrived — compute the lock-on move and transition to TRACK.

All helpers are module-level functions that take the ``DetectWorker`` instance
as their first argument so they can reach the per-thread objects stored on
``self`` without a class hierarchy.
"""

__all__ = ["step_idle"]

import time

from dronetracker.tracking.selection import pick_best_track
from dronetracker.ptz.lockon import compute_lockon_command
from dronetracker.ptz.calibration import update_ego_gain

# EMA + calibration constants (derived from UNV lens — not user-tunable)
_CALIB_EMA    = 0.85
_CALIB_MIN_DP = 2e-4
_CALIB_MIN_N  = 3


def step_idle(worker, fr, fh, fw, now, mag, pan_now, tilt_now,
              skip_calib, do_lock, ctx):
    """Run one IDLE-state frame.  Returns the (possibly updated) ``_DetectCtx``."""
    cfg = worker._cfg

    if now - ctx.last_mask_t > cfg.motion.ground_mask_interval_s:
        worker._gmg.compute(fr)
        ctx.last_mask_t = now

    det_fr      = _apply_osd_mask(fr, cfg)
    motion_mask = worker._mt.detect_motion(det_fr)
    detections  = worker._motion_to_detections(
        motion_mask,
        min_area         = cfg.motion.min_area,
        max_area         = cfg.motion.max_area,
        cluster_distance = cfg.motion.cluster_distance,
    )
    tracked = worker._tracker.update(detections=detections)

    # Streaming track filter
    if cfg.track_filter.enabled:
        if worker._track_filter is None:
            from dronetracker.tracking.filtering import TrackFilter
            worker._track_filter = TrackFilter(fw, fh, cfg.track_filter)
        survivors = worker._track_filter.update(tracked)
        tracked = [o for o in tracked if o.id in survivors]

    # Online ego-motion calibration from arrow-key moves
    if (not skip_calib and ctx.prev_calib_pose is not None
            and worker._mt.last_affine is not None and mag > 0.0):
        ctx = _update_calibration(worker, ctx, pan_now, tilt_now, mag)
    ctx.prev_calib_pose = (pan_now, tilt_now)

    # Auto-engage scoring on the (filtered) survivor tracks — stamps
    # obj.drone_score on each so the published tracks carry it to the renderer.
    auto_id = worker._auto.update(tracked, worker._gmg, now)

    worker._state.publish_det(tracks=list(tracked), locked_id=None)

    sm = worker._state.storage
    if sm is not None:
        sm.add_track_frame(tracked, ctx.frame_index, ctx.locked_id)

    # Manual lock (T/Enter) always takes priority over auto-engage.
    if do_lock:
        # Prefer the pink (auto-hunt would-zoom) track — the highest drone-score
        # candidate, which the evaluator already flagged this frame.  Only a
        # currently-detected one (never a coasting ghost), mirroring the auto path.
        # The score is stamped regardless of cfg.auto_engage.enabled, so this works
        # even with hands-free auto-engage off.  Fall back to the longest-lived /
        # most-central pick when no track is flagged pink.
        pink_obj = next(
            (o for o in tracked
             if getattr(o, "auto_engage", False) and o.last_detection is not None),
            None,
        )
        if pink_obj is not None:
            return _execute_lockon(worker, pink_obj, pink_obj.id, fw, fh, now, ctx)
        best     = pick_best_track(tracked, fw, fh)
        best_obj = next((o for o in tracked if o.id == best), None) if best is not None else None
        if best_obj is not None:
            return _execute_lockon(worker, best_obj, best, fw, fh, now, ctx)
        print("[IDLE] no confirmed tracks to lock onto")
        return ctx

    # Automatic IDLE -> TRACK: engage the best drone-like candidate, unless the
    # feature is disabled or we are still inside the post-unlock cooldown.
    if (cfg.auto_engage.enabled and auto_id is not None
            and now >= ctx.auto_cooldown_until):
        best_obj = next((o for o in tracked if o.id == auto_id), None)
        # Only ever lock onto a currently-detected track (a real, on-screen blob),
        # exactly like the manual path — never a coasting/extrapolated ghost.
        if best_obj is not None and best_obj.last_detection is not None:
            print(f"[AUTO] engaging ID={auto_id} "
                  f"score={getattr(best_obj, 'drone_score', 0.0):.2f}")
            return _execute_lockon(worker, best_obj, auto_id, fw, fh, now, ctx)

    return ctx


# ── Private helpers ────────────────────────────────────────────────────────────

def _execute_lockon(worker, best_obj, track_id, fw, fh, now, ctx):
    """Center+zoom onto ``best_obj`` and transition the state machine to TRACK.

    Shared by the manual (T/Enter) and automatic engage paths so both behave
    identically.  Returns the (updated) ctx; stays IDLE without changing state
    if the PTZ rejects the move twice (busy).
    """
    cfg = worker._cfg
    cmd = compute_lockon_command(
        best_obj, fw, fh, ctx.fps_val, worker._ptz.zoom_pos,
        cfg.frozen, cfg.zoom)

    from dronetracker.ptz.controller import TRANSLATION_SPACE_FOV
    accepted = worker._ptz.center_and_zoom(
        cmd.dx, cmd.dy, cmd.zoom_pos_target,
        space=TRANSLATION_SPACE_FOV,
        parallel=cfg.frozen.zoom_parallel)
    if not accepted:
        time.sleep(0.05)
        accepted = worker._ptz.center_and_zoom(
            cmd.dx, cmd.dy, cmd.zoom_pos_target,
            space=TRANSLATION_SPACE_FOV,
            parallel=cfg.frozen.zoom_parallel)
    if not accepted:
        print("[IDLE] center_and_zoom rejected after retry (PTZ busy); staying IDLE")
        return ctx

    ctx.reset_track_fields()
    ctx.locked_id = track_id
    ctx.lock_t    = now
    ctx.state     = "TRACK"
    worker._state.publish_det(tracks=[])
    print(f"[TRACK] locked ID={track_id}  "
          f"aim=({cmd.aim_x:.0f},{cmd.aim_y:.0f}) ex={cmd.ex:.3f} ey={cmd.ey:.3f}  "
          f"dx={cmd.dx:.3f} dy={cmd.dy:.3f}  lead={cmd.lead_frames:.1f}f v=({cmd.vx:.2f},{cmd.vy:.2f})  "
          f"zoom {cmd.zoom_pos_now:.3f}→{cmd.zoom_pos_target:.3f}")
    return ctx


def _apply_osd_mask(fr, cfg):
    """Return a copy of fr with the OSD timestamp region zeroed out.

    The copy is used only for motion detection — the display frame is
    untouched.  osd_mask_rect = [x, y, w, h]; empty list disables.
    """
    rect = cfg.motion.osd_mask_rect
    if not rect or len(rect) < 4:
        return fr
    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    out = fr.copy()
    out[y:y + h, x:x + w] = 0
    return out


def _update_calibration(worker, ctx, pan_now, tilt_now, mag):
    """EMA-update ego-motion gain from the latest optical-flow affine shift."""
    dpan  = pan_now  - ctx.prev_calib_pose[0]
    dtilt = tilt_now - ctx.prev_calib_pose[1]
    tx = float(worker._mt.last_affine[0, 2])
    ty = float(worker._mt.last_affine[1, 2])

    new_k_pan  = update_ego_gain(ctx.calib_k_pan,  tx, dpan,  mag, _CALIB_EMA, _CALIB_MIN_DP)
    new_k_tilt = update_ego_gain(ctx.calib_k_tilt, ty, dtilt, mag, _CALIB_EMA, _CALIB_MIN_DP)

    if new_k_pan != ctx.calib_k_pan or new_k_tilt != ctx.calib_k_tilt:
        ctx.calib_n = min(ctx.calib_n + 1, 200)
        if ctx.calib_n == _CALIB_MIN_N:
            print(f"[Calib] ego-motion trusted: k_pan={new_k_pan:.1f}  k_tilt={new_k_tilt:.1f}")
    ctx.calib_k_pan  = new_k_pan
    ctx.calib_k_tilt = new_k_tilt
    return ctx
