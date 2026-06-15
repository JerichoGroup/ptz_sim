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

from utils.MotionDetector import MotionDetector
from utils.GroundMaskGenerator import GroundMaskGenerator
MotionDetector.SHOW_WINDOW  = False
GroundMaskGenerator.SHOW_IMAGE = False

from dronetracker.config.loader import load_pipeline_config
from dronetracker.detection.clustering import motion_to_detections
from dronetracker.pipeline.offline import OfflineRunner


def _build_runner(cfg) -> OfflineRunner:
    off = cfg.offline

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=off.distance_threshold,
        hit_counter_max=200,
        initialization_delay=off.initialization_delay,
    )

    # Mutable closure state for the detect function
    _state = {"gmg": None, "mt": None, "fps": 25, "frame_idx": 0}

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

        return motion_to_detections(
            motion_mask,
            min_area=off.min_motion_area,
            max_area=off.max_motion_area,
            cluster_distance=off.cluster_distance,
        )

    return OfflineRunner(
        detect_fn   = detect_fn,
        tracker     = tracker,
        init_fn     = init_fn,
        show_window = off.show_window,
    )


def main():
    cfg    = load_pipeline_config()
    off    = cfg.offline
    runner = _build_runner(cfg)
    runner.run(
        video_path  = off.video_path,
        output_path = off.output_path,
        start_sec   = off.start_sec,
        window_name = "Motion Tracking (Norfair)",
    )


if __name__ == "__main__":
    cProfile.run("main()")
