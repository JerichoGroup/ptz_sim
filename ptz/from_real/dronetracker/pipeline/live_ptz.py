# -*- coding: utf-8 -*-
"""Live PTZ tracking pipeline orchestrator.

Three threads, one clear job each:

  grab thread    — ``RtspGrabber``  (daemon, reads RTSP, writes latest_frame)
  detect thread  — ``DetectWorker`` (daemon, runs motion+YOLO+PTZ state machine)
  main thread    — ``DisplayLoop``  (owns the cv2 window + keyboard)

Cross-thread state lives in ``SharedState``.  All detect-thread logic (motion
detector, tracker, YOLO helpers, IDLE/TRACK state machine) lives in
``DetectWorker`` (``detect.py``); the main-thread display/keyboard loop
lives in ``DisplayLoop`` (``display.py``).  ``LivePtzPipeline`` is the glue that
builds everything and wires the three threads together.
"""

__all__ = ["LivePtzPipeline"]

import threading
import time

from dronetracker.config.schema import PipelineConfig
from dronetracker.pipeline.shared_state import SharedState
from dronetracker.pipeline.detect import DetectWorker
from dronetracker.pipeline.display import DisplayLoop
from dronetracker.detection.classifier import YoloClassifier
from dronetracker.media.rtsp import RtspGrabber


class LivePtzPipeline:
    """Full live PTZ pipeline: grab → detect/control → display."""

    def __init__(self, cfg: PipelineConfig, ptz=None):
        self._cfg = cfg

        # PTZ controller.  Default: construct the real ONVIF controller from
        # config values.  A pre-built controller may be injected instead (e.g.
        # PTZSimController from the simulator entry point) — it must expose the
        # same public API.  Passing one leaves the real-camera path untouched.
        if ptz is None:
            from dronetracker.ptz.controller import PTZController
            ptz = PTZController(
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
        self._ptz = ptz

        self._state = SharedState()

        # Segmented, size-capped SSD storage (no-op when storage.enabled is False).
        if cfg.storage.enabled:
            from dronetracker.media.storage import StorageManager
            self._state.storage = StorageManager(cfg.storage)
            print(f"[Storage] enabled  base_dir={cfg.storage.base_dir}  "
                  f"max={cfg.storage.max_size_gb}GB  "
                  f"segment={cfg.storage.segment_seconds}s  "
                  f"auto_start={cfg.storage.auto_start}", flush=True)
        else:
            print("[Storage] disabled — legacy CWD recording (V key)", flush=True)

        # YOLO classifier (loaded via _classifier.load() before threads start)
        self._classifier = YoloClassifier(cfg.yolo)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Load YOLO, start grab + detect threads, then run the display loop."""
        self._classifier.load()

        grabber = RtspGrabber(
            self._cfg.camera.rtsp_main,
            self._cfg.camera.rtsp_alt,
            self._state,
        )
        grabber.start()

        worker = DetectWorker(self._cfg, self._ptz, self._state, self._classifier)
        threading.Thread(target=worker.run, daemon=True, name="detect").start()

        time.sleep(3)   # let grab + PTZ startup complete
        DisplayLoop(self._cfg, self._ptz, self._state).run()
