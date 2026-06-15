#!/usr/bin/env python3
"""Offline tiled YOLOv11 + Norfair tracking pipeline.

Splits each frame into overlapping tiles, runs YOLO on each, merges via NMS,
then tracks with Norfair.  Writes an annotated video.

Paths and thresholds are currently hard-coded below (same as the original
tiled_tracking.py).  They will be migrated to config.yaml in a future pass.

Usage:
    python dronetracker/apps/offline_tiled.py

Run in background for long videos:
    python dronetracker/apps/offline_tiled.py &
"""

from norfair import Tracker
from ultralytics import YOLO

from utils.tiling import split_into_tiles
from utils.nms import global_nms
from utils.tracking import create_norfair_detections, draw_tracks

from dronetracker.media.video import MP4_FOURCC
from dronetracker.pipeline.offline import OfflineRunner

# ── Config (will move to config.yaml in a future pass) ───────────────────────
VIDEO_PATH  = "/home/guy/experiment_07_05/rgb-01/09_17.mp4"
MODEL_PATH  = "/home/guy/dronesDS/DroneTracker/results/runs/detect/train-15/weights/best.pt"
OUTPUT_PATH = "/home/guy/experiment_07_05/rgb-01/09_17_output.mp4"
DEVICE          = "cuda"
TILE_SIZE       = (640, 640)
CONFIDENCE      = 0.25
IOU_THRESHOLD   = 0.1
SHOW_WINDOW     = True


def _build_runner() -> OfflineRunner:
    model = YOLO(MODEL_PATH)

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=120,
        hit_counter_max=30,
        initialization_delay=2,
    )

    def detect_fn(frame):
        tiles, coords = split_into_tiles(frame, TILE_SIZE)
        all_boxes, all_scores, all_classes = [], [], []

        for tile, (ox, oy) in zip(tiles, coords):
            res = model.predict(tile, conf=CONFIDENCE,
                                device=DEVICE, verbose=False)[0]
            if res.boxes is None:
                continue
            for box, score, cls in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                all_boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
                all_scores.append(float(score))
                all_classes.append(int(cls))

        boxes, scores, _ = global_nms(
            all_boxes, all_scores, all_classes, IOU_THRESHOLD)
        return create_norfair_detections(boxes, scores)

    def draw_fn(frame, tracked_objects):
        return draw_tracks(frame, tracked_objects)

    return OfflineRunner(
        detect_fn   = detect_fn,
        tracker     = tracker,
        draw_fn     = draw_fn,
        fourcc      = MP4_FOURCC,
        show_window = SHOW_WINDOW,
    )


def main():
    runner = _build_runner()
    runner.run(
        video_path  = VIDEO_PATH,
        output_path = OUTPUT_PATH,
        window_name = "Tiled YOLOv11 Tracking",
    )


if __name__ == "__main__":
    main()
