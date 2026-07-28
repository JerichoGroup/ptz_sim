# -*- coding: utf-8 -*-
"""Tiled YOLO classifier for the live PTZ pipeline.

``YoloClassifier`` owns the YOLO model and all of its runtime knobs (device,
half precision).  ``load()`` builds + warms up the model (and, for a TensorRT
``.engine``, the CUDA context for the runtime batch size) and must be called on
the main thread before the detect thread starts.  ``classify()`` is then called
only from the detect thread.

The class carries no ``SharedState`` reference — the detect-loop FPS shown in the
``[YOLO]`` log line is passed in as ``loop_fps`` so the classifier stays
trivially testable.  Heavy imports (torch / ultralytics) are lazy to preserve
Jetson startup timing.
"""

__all__ = ["YoloClassifier"]

import time

import numpy as np

from dronetracker.vision.tiling import split_into_tiles
from dronetracker.vision.nms import global_nms


class YoloClassifier:
    """Load + warm up a YOLO model and run tiled inference with global NMS."""

    # IoU threshold for merging detections across overlapping tiles.
    _NMS_IOU = 0.4

    def __init__(self, cfg):
        """cfg is a ``YoloConfig`` (model_path, imgsz, conf, warmup_tiles, ...)."""
        self._cfg    = cfg
        self._model  = None
        self._device = "cpu"
        self._half   = False

    # ── Startup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load and warm up the YOLO model (call before starting threads).

        Warmup uses a batch of ``warmup_tiles`` zero images so that a TensorRT
        engine's CUDA context is fully built for the runtime batch size before the
        first live classify.  8 tiles matches a 1920×1080 stream @640 px tiles.
        """
        import torch
        from ultralytics import YOLO
        cfg = self._cfg
        self._device = 0 if torch.cuda.is_available() else "cpu"
        self._half   = torch.cuda.is_available()
        self._model  = YOLO(cfg.model_path, task="detect")

        backend = "tensorrt" if cfg.model_path.endswith(".engine") else "pytorch"
        warmup_batch = [np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8)] * cfg.warmup_tiles
        self._model(
            warmup_batch,
            imgsz=cfg.imgsz, conf=cfg.conf,
            device=self._device, half=self._half,
            verbose=False,
        )
        print(f"[Model] {cfg.model_path}  backend={backend}  GPU:{torch.cuda.is_available()}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def classify(self, frame, loop_fps: float):
        """Run tiled YOLO (tiles + whole frame) with global NMS.

        Returns [(x1,y1,x2,y2,conf,label), ...] in original-frame coordinates.

        ``loop_fps`` is only used for the ``[YOLO]`` diagnostics line.
        """
        t0  = time.perf_counter()
        cfg = self._cfg
        sz  = cfg.imgsz
        tiles, tile_coords = split_into_tiles(frame, (sz, sz))
        batch   = list(tiles)
        coords  = list(tile_coords)
        batch.append(frame)      # whole frame — YOLO handles letterbox internally
        coords.append(None)      # sentinel for "no offset"

        if not batch:
            return []

        t_inf0 = time.perf_counter()
        results = self._model(
            batch,
            imgsz=sz, conf=cfg.conf,
            device=self._device, half=self._half,
            verbose=False,
        )
        t_inf1 = time.perf_counter()

        all_boxes, all_scores, all_classes = [], [], []
        for res, coord in zip(results, coords):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes  = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            if coord is not None:
                ox, oy = coord
                boxes[:, [0, 2]] += ox
                boxes[:, [1, 3]] += oy
            all_boxes.extend(boxes.tolist())
            all_scores.extend(scores.tolist())
            all_classes.extend(classes.tolist())

        dt_total = (time.perf_counter() - t0) * 1e3
        dt_infer = (t_inf1 - t_inf0) * 1e3

        if not all_boxes:
            print(f"[YOLO] total={dt_total:6.1f}ms  infer={dt_infer:6.1f}ms  "
                  f"tiles={len(batch)}  dets=0  loop_fps={loop_fps:.1f}")
            return []

        kept_boxes, kept_scores, kept_classes = global_nms(
            all_boxes, all_scores, all_classes, self._NMS_IOU)
        names = self._model.names or {}
        out = [
            (int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s), names.get(c, "drone"))
            for b, s, c in zip(kept_boxes, kept_scores, kept_classes)
        ]
        print(f"[YOLO] classify={dt_total:6.1f}ms  infer={dt_infer:6.1f}ms  "
              f"tiles={len(batch)}  dets={len(out)}  loop_fps={loop_fps:.1f}")
        return out
