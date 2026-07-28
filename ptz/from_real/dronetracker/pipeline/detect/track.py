# -*- coding: utf-8 -*-
"""TRACK state step for the live PTZ detect thread.

``step_track`` runs once per frame while the state machine is in TRACK.  It
manages three sequential phases:

  Phase A — Record the AF-start moment (first frame after zoom clears).
  Phase B — Sharpness-gated classify: wait until the lens converges (Laplacian
             variance plateau or timeout), then run tiled YOLO + save crops.
  Monitor  — Post-classify throttled YOLO monitoring + continuous follow loop.
             When ``frozen.follow_enabled`` is True, ``_follow_step`` runs every
             frame to drive pan/tilt via PID; YOLO updates the prediction filter
              on its throttled cadence.  Auto-unlock after ``frozen.track_miss_frames``
             consecutive misses.

Returns ``True`` when TRACK should exit (i.e. the caller should call
``_reset_to_idle``).
"""

__all__ = ["step_track"]

from dronetracker.vision.focus import center_gray_roi, roi_sharpness
from dronetracker.media.captures import save_classified_crops




def _log_detections(worker, ctx, dets) -> None:
    """Append this frame's YOLO detections to the current chunk's detections.json."""
    sm = worker._state.storage
    if sm is None or not dets:
        return
    for box in dets:
        sm.add_detection(box, ctx.frame_index)


def _save_crops(worker, fr, dets) -> None:
    """Write classified crops to the current chunk's crops/ dir (legacy: CWD)."""
    sm = worker._state.storage
    if sm is None:
        save_classified_crops(fr, dets)      # legacy: current working directory
        return
    crops_dir = sm.crops_dir()               # None when save_crops is disabled
    if crops_dir is not None:
        save_classified_crops(fr, dets, out_dir=crops_dir)


def _select_followed_box(dets, follower, fw, fh, gate_px, now):
    """Return the detection nearest to the follower's current prediction.

    Parameters
    ----------
    dets     : list of (x1, y1, x2, y2, conf, label)
    follower : TargetFollower instance
    fw, fh   : frame width / height in pixels
    gate_px  : max pixel distance from prediction to accept a box
    now      : current wall-clock time (used for velocity extrapolation)

    Returns
    -------
    The (x1, y1, x2, y2, conf, label) tuple nearest to the prediction, or
    None if dets is empty or the nearest centre exceeds gate_px.
    """
    if not dets:
        return None

    # Reference point: current follower position (lead_s=0) or frame centre.
    # Pass `now` so velocity-based extrapolation since the last measurement is applied.
    if follower.initialized:
        ref_x, ref_y = follower.predict(now, lead_s=0, fw=fw, fh=fh)
    else:
        ref_x, ref_y = fw / 2.0, fh / 2.0

    best_det  = None
    best_dist = float("inf")
    for det in dets:
        cx = (det[0] + det[2]) / 2.0
        cy = (det[1] + det[3]) / 2.0
        dist = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_det  = det

    # When the follower has no prior position, accept any detection so the
    # first monitor hit bootstraps the filter regardless of where in the frame
    # the target appears.  Only apply the gate once we have a valid estimate.
    if not follower.initialized:
        return best_det
    return best_det if best_dist <= gate_px else None


def _follow_step(worker, ctx, now, fr):
    """Predict target, run PID, and issue a move() command.

    Called every frame after classify_done, before the YOLO throttle guard.
    No-ops while the camera is mid-move (RelativeMove or zoom burst).
    """
    cfg = worker._cfg
    # Skip while camera is executing a move — avoid chasing a moving reference.
    if worker._ptz._rel_moving or worker._ptz._zooming:
        return
    fh, fw = fr.shape[:2]
    px, py = worker._follower.predict(now, cfg.frozen.follow_lead_s, fw, fh)
    cx_n = px / fw
    cy_n = py / fh
    vx, vy = worker._pid.compute(cx_n, cy_n)
    # Apply sign flips (field-correctable without touching pid.py)
    vx *= cfg.frozen.follow_sign_x
    vy *= cfg.frozen.follow_sign_y
    # Apply vel_scale so fast panning at tele zoom doesn't overshoot
    if cfg.frozen.follow_apply_vel_scale:
        s = worker._ptz.vel_scale()
        vx *= s
        vy *= s
    # Coast guard: stop issuing move() if we've lost the target too long
    if worker._follower.coast_n > cfg.frozen.follow_coast_frames:
        return   # camera holds; mover auto-stops on cmd_ttl staleness
    worker._ptz.move(vx, vy)


def step_track(worker, ctx, now, fr) -> bool:
    """Run one TRACK-state frame.  Returns True if TRACK should exit."""
    cfg = worker._cfg
    worker._state.publish_det(state="TRACK", locked_id=ctx.locked_id)

    # Phase A — record the moment TRACK starts (first frame after zoom clears).
    # step_track is only reached after _ptz._zooming clears, so the zoom burst
    # has already finished and center_and_zoom() has already issued one autofocus_once()
    # call.  We do NOT fire a second AF here — that was the source of the double-hunt.
    if not ctx.focus_done:
        ctx.focus_done  = True
        ctx.focus_t     = now
        ctx.focus_best  = -1.0
        ctx.focus_stale = 0
        print("[TRACK] AF settling (issued post-zoom)")
        return False

    # Phase B — sharpness-gated classify.
    if ctx.focus_done and not ctx.classify_done:
        gray_roi  = center_gray_roi(fr, cfg.frozen.focus_roi_frac)
        sharpness = roi_sharpness(gray_roi)
        if sharpness > ctx.focus_best:
            ctx.focus_best  = sharpness
            ctx.focus_stale = 0
        else:
            ctx.focus_stale += 1

        pan_tilt_settled = (now - ctx.lock_t)  > cfg.frozen.move_settle_s
        af_converged     = ctx.focus_stale      >= cfg.frozen.focus_plateau_frames
        timed_out        = (now - ctx.focus_t)  > cfg.frozen.focus_settle_s

        if not (pan_tilt_settled and (af_converged or timed_out)):
            return False

        if timed_out and not af_converged:
            print(f"[TRACK] AF timeout after {now - ctx.focus_t:.2f}s — classify anyway")
        else:
            print(f"[TRACK] AF converged in {now - ctx.focus_t:.2f}s  "
                  f"sharpness={ctx.focus_best:.1f}  stale={ctx.focus_stale}")

        dets = worker._classifier.classify(fr, worker._state.fps)
        if dets:
            best = max(dets, key=lambda d: d[4])
            if cfg.frozen.follow_enabled:
                # Bootstrap the follower on the dominant box and mark it followed.
                fh, fw = fr.shape[:2]
                cx = (best[0] + best[2]) / 2.0
                cy = (best[1] + best[3]) / 2.0
                worker._follower.measure(cx, cy, now)
                aim = worker._follower.predict(now, cfg.frozen.follow_lead_s, fw, fh)
                worker._state.set_track_overlay(best, followed=True, aim=aim)
            else:
                worker._state.set_track_overlay(best, followed=False, aim=None)
                ctx.last_track_cx = (best[0] + best[2]) / 2.0
                ctx.last_track_cy = (best[1] + best[3]) / 2.0
        else:
            worker._state.set_track_overlay(None, followed=False, aim=None)
        worker._state.set_yolo_boxes(dets)
        _log_detections(worker, ctx, dets)
        _save_crops(worker, fr, dets)
        worker._state.publish_det(request_shot=True)
        ctx.classify_done  = True
        ctx.last_monitor_t = now
        ctx.miss_frames    = 0 if dets else 1
        print(f"[TRACK] classify done — {len(dets)} detection(s)")
        return False

    # Monitor — throttled YOLO after classify + continuous follow loop.
    if ctx.classify_done:

        # Follow step runs EVERY frame (predict → PID → move), regardless of YOLO throttle.
        if cfg.frozen.follow_enabled:
            _follow_step(worker, ctx, now, fr)

        # YOLO monitoring is still throttled.
        if (now - ctx.last_monitor_t) < cfg.frozen.monitor_interval_s:
            return False
        ctx.last_monitor_t = now

        dets = worker._classifier.classify(fr, worker._state.fps)
        _log_detections(worker, ctx, dets)

        if cfg.frozen.follow_enabled:
            # A "miss" is losing the FOLLOWED TARGET — not the mere absence of
            # detections.  Clutter (e.g. a passing bird) must not reset the
            # unlock counter, or the camera would coast forever and never home.
            fh, fw = fr.shape[:2]
            followed = (_select_followed_box(
                dets, worker._follower, fw, fh, cfg.frozen.follow_gate_px, now
            ) if dets else None)
            if followed is not None:
                worker._state.set_yolo_boxes(dets)
                ctx.miss_frames = 0
                cx = (followed[0] + followed[2]) / 2.0
                cy = (followed[1] + followed[3]) / 2.0
                worker._follower.measure(cx, cy, now)
                aim = worker._follower.predict(now, cfg.frozen.follow_lead_s, fw, fh)
                # Display the followed target (red FOLLOW) + aim crosshair.
                worker._state.set_track_overlay(followed, followed=True, aim=aim)
            else:
                # Target not seen this cycle (no dets, or all gated out as clutter).
                ctx.miss_frames += 1
                worker._follower.coast()
                aim = (worker._follower.predict(now, cfg.frozen.follow_lead_s, fw, fh)
                       if worker._follower.initialized else None)
                worker._state.set_track_overlay(None, followed=False, aim=aim)
                if ctx.miss_frames >= cfg.frozen.track_miss_frames:
                    print(f"[TRACK] target lost for {cfg.frozen.track_miss_frames} checks → IDLE")
                    return True
        else:
            # Follow off: gate detections by distance from last known position,
            # so clutter (birds, clouds) doesn't reset the unlock counter.
            if dets:
                fh, fw = fr.shape[:2]
                best_dist = float("inf")
                best_det  = None
                for d in dets:
                    cx = (d[0] + d[2]) / 2.0
                    cy = (d[1] + d[3]) / 2.0
                    dist = ((cx - ctx.last_track_cx)**2 + (cy - ctx.last_track_cy)**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_det  = d
                if best_dist <= cfg.frozen.follow_gate_px:
                    worker._state.set_yolo_boxes(dets)
                    ctx.miss_frames = 0
                    ctx.last_track_cx = (best_det[0] + best_det[2]) / 2.0
                    ctx.last_track_cy = (best_det[1] + best_det[3]) / 2.0
                    best = max(dets, key=lambda d: d[4])
                    worker._state.set_track_overlay(best, followed=False, aim=None)
                else:
                    ctx.miss_frames += 1
                    worker._state.clear_yolo_all()
                    if ctx.miss_frames >= cfg.frozen.track_miss_frames:
                        print(f"[TRACK] target lost for {cfg.frozen.track_miss_frames} checks → IDLE")
                        return True
            else:
                ctx.miss_frames += 1
                worker._state.clear_yolo_all()
                if ctx.miss_frames >= cfg.frozen.track_miss_frames:
                    print(f"[TRACK] no detection for {cfg.frozen.track_miss_frames} checks → IDLE")
                    return True

    return False
