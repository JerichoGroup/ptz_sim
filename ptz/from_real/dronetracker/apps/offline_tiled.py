#!/usr/bin/env python3
"""Offline tiled YOLOv11 + Norfair tracking pipeline.

Splits each frame into overlapping tiles, runs YOLO on each, merges via NMS,
then tracks with Norfair.  Writes an annotated video.

Paths and thresholds are currently hard-coded below.
They will be migrated to config.yaml in a future pass.

Usage:
    python dronetracker/apps/offline_tiled.py

Run in background for long videos:
    python dronetracker/apps/offline_tiled.py &
"""

from norfair import Tracker
from ultralytics import YOLO

from dronetracker.media.video import AVI_FOURCC
from dronetracker.pipeline.offline import OfflineRunner
from dronetracker.rendering.draw import draw_yolo_box

# ── Config (will move to config.yaml in a future pass) ───────────────────────
VIDEO_PATH  = "/home/guy/dronesDS/experiment_18_06/18_06_2026__11_15_recovered.mp4"
MODEL_PATH  = "/home/guy/Downloads/yolo_training/yolo11s-p2-merged/weights/best.pt"
OUTPUT_PATH = "/home/guy/dronesDS/experiment_18_06/11_15_355_400s_11s.avi"
START_SEC       = 235.0
END_SEC         = 240.0
DEVICE          = "cuda"
CONFIDENCE      = 0.25
SHOW_WINDOW     = False


def _build_runner() -> OfflineRunner:
    model = YOLO(MODEL_PATH)
    class_names = model.names  # {idx: label}

    # Norfair tracker is required by OfflineRunner but unused — YOLO's built-in
    # ByteTrack (persist=True) handles tracking; detect_fn returns no detections.
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=120,
        hit_counter_max=1,
        initialization_delay=0,
    )

    _state = {"yolo_boxes": []}

    def detect_fn(frame):
        res = model.track(frame, persist=True, conf=CONFIDENCE,
                          device=DEVICE, verbose=False)[0]
        boxes = []
        if res.boxes is not None and len(res.boxes):
            for box, score, cls in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                boxes.append((x1, y1, x2, y2,
                               float(score),
                               class_names.get(int(cls), str(cls))))
        _state["yolo_boxes"] = boxes
        return []  # no Norfair detections

    def draw_fn(frame, tracked_objects):
        for yolo_box in _state["yolo_boxes"]:
            draw_yolo_box(frame, yolo_box, sx=1.0, sy=1.0)
        return frame

    return OfflineRunner(
        detect_fn   = detect_fn,
        tracker     = tracker,
        draw_fn     = draw_fn,
        fourcc      = AVI_FOURCC,
        show_window = SHOW_WINDOW,
    )


def main():
    runner = _build_runner()
    runner.run(
        video_path  = VIDEO_PATH,
        output_path = OUTPUT_PATH,
        start_sec   = START_SEC,
        end_sec     = END_SEC,
        window_name = "Tiled YOLOv11 Tracking",
    )


if __name__ == "__main__":
    main()
