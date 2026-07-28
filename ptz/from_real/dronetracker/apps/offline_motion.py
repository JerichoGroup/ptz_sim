#!/usr/bin/env python3
"""Offline motion-detection + Norfair tracking pipeline.

Reads a video, runs the stabilised frame-diff detector (MotionDetector +
GroundMaskGenerator), clusters motion blobs, tracks with Norfair, and
writes an annotated AVI.

Configuration is loaded from config.yaml / .env (see config.example.yaml).
Hardcoded paths and thresholds now live under the ``offline:`` section.

Usage:
    python dronetracker/apps/offline_motion.py

Run in background for long videos:
    python dronetracker/apps/offline_motion.py &
"""

import cProfile

from norfair import Tracker

from dronetracker.vision.motion_detector import MotionDetector
from dronetracker.vision.ground_mask import GroundMaskGenerator
MotionDetector.SHOW_WINDOW  = False
GroundMaskGenerator.SHOW_IMAGE = False

from dronetracker.config.loader import load_pipeline_config
from dronetracker.detection.clustering import motion_to_detections
from dronetracker.pipeline.offline import OfflineRunner
from dronetracker.tracking.auto_engage import AutoEngageEvaluator, FeatureLogger
from dronetracker.tracking.distance import make_motion_size_distance, make_reid_distance
from dronetracker.tracking.appearance import AppearanceStore, compute_descriptor, blob_contrast
from dronetracker.tracking.template_track import TemplateTracker


def _build_runner(cfg) -> OfflineRunner:
    off = cfg.offline

    # Per-track appearance template store (soft-gate ID protection).  Only
    # active when both the motion-size distance and appearance are enabled.
    use_appearance = off.use_motion_size_distance and off.use_appearance
    # Templates must outlive the ReID window so a dead mature track can still be
    # reclaimed by appearance.
    store = (AppearanceStore(alpha=off.appearance_ema_alpha,
                             prune_after=off.reid_hit_counter_max + 300)
             if use_appearance else None)
    appearance_weight = off.appearance_weight if use_appearance else 0.0

    if off.use_motion_size_distance:
        dist_fn = make_motion_size_distance(
            size_weight=off.size_weight,
            size_penalty_cap=off.size_penalty_cap,
            appearance_store=store,
            appearance_weight=appearance_weight,
            appearance_deadband=off.appearance_deadband,
            appearance_penalty_cap=off.appearance_penalty_cap,
            appearance_maturity_frames=off.appearance_maturity_frames,
            appearance_dormancy_frames=off.appearance_dormancy_frames,
            appearance_dormant_cap=off.appearance_dormant_cap,
            appearance_slow_speed=off.appearance_slow_speed,
        )
    else:
        dist_fn = "euclidean"

    tracker_kwargs = dict(
        distance_function   = dist_fn,
        distance_threshold  = off.distance_threshold,
        hit_counter_max     = off.hit_counter_max,      # was hard-coded 200; now from config
        initialization_delay = off.initialization_delay,
    )
    if off.use_motion_size_distance and off.reid_enabled:
        tracker_kwargs["reid_distance_function"]  = make_reid_distance(
            size_weight=off.size_weight,
            size_penalty_cap=off.size_penalty_cap,
            appearance_store=store,
            appearance_weight=appearance_weight,
            appearance_deadband=off.appearance_deadband,
            appearance_penalty_cap=off.appearance_penalty_cap,
            appearance_maturity_frames=off.appearance_maturity_frames,
            appearance_dormant_cap=off.appearance_dormant_cap,
        )
        tracker_kwargs["reid_distance_threshold"] = off.reid_distance_threshold
        tracker_kwargs["reid_hit_counter_max"]    = off.reid_hit_counter_max

    tracker = Tracker(**tracker_kwargs)

    # Mutable closure state for the detect function
    _state = {"gmg": None, "mt": None, "fps": 25, "frame_idx": 0, "prev_frame": None}

    def init_fn(first_frame, fps, width, height):
        _state["fps"] = fps
        gmg = GroundMaskGenerator()
        gmg.compute(first_frame)
        mt  = MotionDetector(ground_mask_generator=gmg)
        _state["gmg"] = gmg
        _state["mt"]  = mt

    def detect_fn(frame):
        mt  = _state["mt"]
        gmg = _state["gmg"]
        fps = _state["fps"]
        idx = _state["frame_idx"]
        _state["frame_idx"] += 1

        mask_interval = fps * off.ground_mask_interval_s
        if mask_interval > 0 and idx % int(mask_interval) == 0:
            gmg.compute(frame)

        was_moving  = mt.camera_moving
        motion_mask = mt.detect_motion(frame)
        if was_moving and not mt.camera_moving:
            gmg.compute(frame)

        # The motion blob trails the object by ~1 frame (frame-diff + temporal
        # accumulation), so sample appearance from the PREVIOUS frame, masked to
        # the motion pixels.  No previous frame yet (first call) → skip appearance.
        prev = _state["prev_frame"]
        _state["prev_frame"] = frame

        appearance_fn = None
        contrast_fn = None
        sample_frame = None
        if prev is not None:
            sample_frame = prev
            if use_appearance:
                appearance_fn = lambda f, box, m: compute_descriptor(
                    f, box, mask=m,
                    min_box_px=off.appearance_min_box_px,
                    min_pixels=off.appearance_min_pixels,
                )
            if cfg.track_filter.min_blob_contrast > 0:
                contrast_fn = lambda f, box, m: blob_contrast(f, box, m)

        return motion_to_detections(
            motion_mask,
            min_area=off.min_motion_area,
            max_area=off.max_motion_area,
            cluster_distance=off.cluster_distance,
            frame=(sample_frame if (appearance_fn or contrast_fn) else None),
            appearance_fn=appearance_fn,
            contrast_fn=contrast_fn,
        )

    ae        = cfg.auto_engage
    ae_logger = FeatureLogger(ae.log_path) if ae.log_enabled else None
    evaluator = AutoEngageEvaluator(ae, logger=ae_logger)

    # Template-at-location tracker for mature, near-stationary objects.
    tt = (TemplateTracker(
            max_speed=off.template_max_speed,
            search_radius=off.template_search_radius,
            match_threshold=off.template_match_threshold,
            pad=off.template_pad,
            suppress_radius=off.template_suppress_radius,
            maturity_frames=off.appearance_maturity_frames,
            appearance_store=store,
            match_dist=off.template_match_dist,
          )
          if off.template_track_enabled else None)

    def on_video_start():
        if store is not None:
            store.clear()
        if tt is not None:
            tt.clear()

    return OfflineRunner(
        detect_fn        = detect_fn,
        tracker          = tracker,
        init_fn          = init_fn,
        show_window      = off.show_window,
        track_filter_cfg = cfg.track_filter,
        evaluator        = evaluator,
        after_update     = store.update if store is not None else None,
        on_video_start   = on_video_start,
        augment_fn       = tt.augment  if tt is not None else None,
        observe_fn       = tt.observe  if tt is not None else None,
    )


def main():
    cfg    = load_pipeline_config()
    off    = cfg.offline
    runner = _build_runner(cfg)
    runner.run(
        video_path  = off.video_path,
        output_path = off.output_path,
        start_sec   = off.start_sec,
        end_sec     = off.end_sec if off.end_sec > 0 else None,
        window_name = "Motion Tracking (Norfair)",
    )


if __name__ == "__main__":
    cProfile.run("main()")
