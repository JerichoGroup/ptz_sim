# -*- coding: utf-8 -*-
"""Main-thread display loop for the live PTZ pipeline.

``DisplayLoop`` owns the cv2 window and everything that runs on the main thread:
resize the latest frame, draw track/YOLO overlays + HUD, write the recording,
``imshow``, handle keyboard input, and tear everything down on quit.  It reads
detect-side state through ``SharedState`` and drives the camera through the
shared ``PTZController``; it never touches ``DetectWorker`` directly.

Constructed with ``(cfg, ptz, state)`` and run via ``run()`` from the main
thread.  Heavy imports (cv2) stay lazy.
"""

__all__ = ["DisplayLoop"]

import math
import threading
import time
from datetime import datetime

import numpy as np

from dronetracker.config.schema import PipelineConfig
from dronetracker.pipeline.shared_state import SharedState
from dronetracker.rendering.hud import HudState, draw_hud
from dronetracker.rendering.draw import draw_tracks_overlay, draw_yolo_box, draw_aim_crosshair
from dronetracker.media.video import make_writer
from dronetracker.media.captures import save_frame


class DisplayLoop:
    """Main thread: resize, overlay, HUD, imshow, keyboard, recording, teardown."""

    def __init__(self, cfg: PipelineConfig, ptz, state: SharedState):
        self._cfg   = cfg
        self._ptz   = ptz
        self._state = state
        # Dual-writer state: separate recordings for IDLE (zoomed-out) and
        # TRACK (zoomed-in).  On a state transition the current writer is
        # closed and a fresh one opened for the new state.
        self._prev_state: str = "IDLE"
        self._writer_idle  = None   # cv2.VideoWriter or None
        self._writer_track = None   # cv2.VideoWriter or None

    # ── Main loop ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main thread: resize, overlay, HUD, imshow, keyboard."""
        import cv2
        from dronetracker.ptz.transform import _F_WIDE, _F_TELE
        from dronetracker.tracking.trails import track_history, prune_stale

        cfg = self._cfg
        WIN = "UAV Motion Tracker — UNV IPC6852ER-X45-VF"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, cfg.display.win_w, cfg.display.win_h)

        # Auto-start segmented recording onto the SSD (no-op without storage).
        if self._state.storage is not None and cfg.storage.auto_start:
            self._rotate_storage_segment()

        dfps = 0.0; d_t = time.time(); d_n = 0
        diag_t = time.time()   # throttle for per-second diagnostics print
        cam_w = cam_h = 0

        while True:
            raw = self._state.get_frame()

            if raw is None:
                raw_disp = None
                disp = np.zeros((cfg.display.win_h, cfg.display.win_w, 3), np.uint8)
                cv2.putText(disp, "Connecting...",
                            (cfg.display.win_w // 2 - 160, cfg.display.win_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
                snap = {"state": "IDLE", "tracks": [], "locked_id": None,
                        "yolo_box": None, "yolo_boxes": [], "fps": 0.0,
                        "yolo_box_followed": False, "follow_aim": None,
                        "request_shot": False}
            else:
                cam_h, cam_w = raw.shape[:2]
                disp = cv2.resize(raw.copy(), (cfg.display.win_w, cfg.display.win_h))
                raw_disp = disp.copy()  # clean frame before overlays — for recording
                sx = cfg.display.win_w / cam_w
                sy = cfg.display.win_h / cam_h
                snap = self._state.snapshot_for_display()

                now_draw = time.time()
                prune_stale(now_draw, cfg.tracker.trail_max_age_s)
                draw_tracks_overlay(
                    disp, snap["tracks"], snap["locked_id"], sx, sy,
                    track_history, now=now_draw,
                )
                # Follower's predicted aim crosshair (red) — shown even while
                # coasting with no detection, so you can see what the camera chases.
                if snap["follow_aim"] is not None:
                    draw_aim_crosshair(disp, snap["follow_aim"], sx, sy)
                # Draw the actively-followed target (red FOLLOW).
                if snap["yolo_box"] is not None and snap["yolo_box_followed"]:
                    draw_yolo_box(disp, snap["yolo_box"], sx, sy, followed=True)
                # Draw all YOLO detections (cyan YOLO) — shows every model output.
                for yb in snap["yolo_boxes"]:
                    if snap["yolo_box"] is None or yb != snap["yolo_box"]:
                        draw_yolo_box(disp, yb, sx, sy, followed=False)

            # Display FPS
            d_n += 1
            now_d = time.time()
            if now_d - d_t >= 1.0:
                dfps = d_n / (now_d - d_t); d_n = 0; d_t = now_d

            # Per-second diagnostics — thread count + FPS to stderr
            if now_d - diag_t >= 1.0:
                tc = threading.active_count()
                names = sorted(t.name for t in threading.enumerate())
                print(f"[Diag] threads={tc}  dfps={dfps:.0f}  det_fps={snap['fps']:.0f}"
                      f"  state={snap['state']}  names={names}", flush=True)
                diag_t = now_d

            with self._state.det_lock:
                cur_state = self._state.state

            f_mm = self._ptz.zoom_pos * (_F_TELE - _F_WIDE) + _F_WIDE
            fov  = 2 * math.degrees(math.atan(7.18 / (2 * f_mm)))
            hud  = HudState(
                state        = snap["state"],
                locked_id    = snap["locked_id"],
                dfps         = dfps,
                det_fps      = snap["fps"],
                zoom_pos     = self._ptz.zoom_pos,
                focal_mm     = f_mm,
                fov_deg      = fov,
                yolo_active  = snap["yolo_box"] is not None,
                cam_w        = cam_w,
                cam_h        = cam_h,
                recording    = self._state.recording,
                ptz_ready    = self._ptz.ready,
                thread_count = threading.active_count(),
                ptz_error    = self._ptz.connect_error,
                auto_engage  = cfg.auto_engage.enabled,
            )
            draw_hud(disp, hud)

            # Rotate to a fresh chunk on schedule.  Also auto-retry when the
            # writer previously failed to open (recording=False, writer=None):
            # should_rotate() uses _segment_start so the retry is naturally
            # spaced by segment_seconds — no spam.  Manual pause is excluded
            # because a paused recording keeps writers alive (not None).
            if (self._state.storage is not None
                    and self._state.storage.should_rotate(now_d)
                    and (self._state.recording
                         or (self._writer_idle is None and self._writer_track is None))):
                self._rotate_storage_segment()

            # Detect IDLE ↔ TRACK transitions and split the recording file.
            if self._state.recording and cur_state != self._prev_state:
                self._rotate_writer_for_state(cur_state)
            self._prev_state = cur_state

            # Write the raw frame to the writer matching the current state.
            if self._state.recording and raw_disp is not None:
                w = self._writer_track if cur_state == "TRACK" else self._writer_idle
                if w is not None:
                    w.write(raw_disp)

            cv2.imshow(WIN, disp)

            if self._state.take_shot_request() and raw_disp is not None:
                fn = save_frame(raw_disp, "freeze")
                print(f"[TRACK] saved {fn}")

            key_raw = cv2.waitKey(1)
            key     = key_raw & 0xFF
            if key == ord("s") and raw_disp is not None:
                self._handle_manual_screenshot(raw_disp)
            elif self._handle_key(key, key_raw, cur_state):
                break

        self._state.stop_ev.set()
        self._ptz.stop_all()
        self._release_all_writers()
        if self._state.storage is not None:
            self._state.storage.close()   # flush the current chunk's JSON logs
        cv2.destroyAllWindows()
        print("[Main] stopped.")

    # ── Keyboard ──────────────────────────────────────────────────────────────

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
        fn = save_frame(disp, "shot")
        print(f"[Shot] {fn}")

    # ── Recording ───────────────────────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        """Toggle recording on/off.

        With segmented SSD storage enabled, 'v' pauses/resumes writing frames
        into the *current* chunk (the chunk and its JSON logs stay open).
        Without storage, it falls back to the legacy behaviour: start/stop a
        timestamped state-prefixed ``.avi`` in the current working directory.

        Recording is split by state: the writer for the current state is opened
        on start, and state transitions (handled elsewhere) close one and open
        the other.
        """
        cfg = self._cfg

        if self._state.storage is not None:
            if self._state.recording:
                self._state.recording = False
                print("[Rec] paused")
            else:
                if self._writer_idle is None and self._writer_track is None:
                    self._rotate_storage_segment()   # opens chunk + writer, sets recording
                else:
                    self._state.recording = True
                if self._state.recording:
                    print("[Rec] resumed")
            return

        # Legacy mode (no storage): state-prefixed .avi in CWD.
        if not self._state.recording:
            self._open_legacy_writer(self._prev_state)
        else:
            self._state.recording = False
            self._release_all_writers()
            print("[Rec] stopped")

    def _rotate_storage_segment(self) -> None:
        """Start a fresh storage chunk: close old writers, open a new one, trim.

        Releases both video writers (finalising the outgoing chunk's AVIs),
        asks the StorageManager to begin a new timestamped chunk (which flushes
        the outgoing chunk's JSON logs), opens a writer for the current state
        inside the new chunk, then enforces the rolling size cap.
        """
        cfg = self._cfg
        sm  = self._state.storage
        self._release_all_writers()
        try:
            sm.start_segment()
        except OSError as exc:
            print(f"[Storage] ERROR: could not create chunk dir: {exc}", flush=True)
            return
        self._open_storage_writer(self._prev_state)
        sm.enforce_quota()

    # ── Dual-writer helpers ───────────────────────────────────────────────────

    def _open_storage_writer(self, state: str) -> None:
        """Open a new video writer for *state* inside the current storage chunk."""
        cfg = self._cfg
        sm  = self._state.storage
        seg_dir = sm.video_path().parent  # chunk directory
        prefix = "track" if state == "TRACK" else "idle"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vp = seg_dir / f"{prefix}_{ts}.avi"
        writer = make_writer(str(vp), 25.0, cfg.display.win_w, cfg.display.win_h)
        if not writer.isOpened():
            print(f"[Storage] ERROR: VideoWriter could not open {vp} "
                  f"— check XVID codec (apt install ffmpeg)", flush=True)
            return
        if state == "TRACK":
            self._writer_track = writer
        else:
            self._writer_idle = writer
        self._state.recording = True
        print(f"[Rec] → {vp}", flush=True)

    def _open_legacy_writer(self, state: str) -> None:
        """Open a state-prefixed .avi in CWD (legacy mode, no StorageManager)."""
        import os
        cfg = self._cfg
        prefix = "track" if state == "TRACK" else "idle"
        fn = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S.avi")
        writer = make_writer(fn, 25.0, cfg.display.win_w, cfg.display.win_h)
        if not writer.isOpened():
            print(f"[Rec] ERROR: VideoWriter could not open {os.path.abspath(fn)} "
                  f"— check XVID codec (apt install ffmpeg)", flush=True)
            return
        if state == "TRACK":
            self._writer_track = writer
        else:
            self._writer_idle = writer
        self._state.recording = True
        print(f"[Rec] → {os.path.abspath(fn)}", flush=True)

    def _rotate_writer_for_state(self, new_state: str) -> None:
        """Close the writer for the old state, open one for the new state.

        Called on IDLE↔TRACK transitions while recording is active.
        """
        # Close the outgoing state's writer.
        if new_state == "TRACK":
            if self._writer_idle is not None:
                self._writer_idle.release()
                self._writer_idle = None
        else:
            if self._writer_track is not None:
                self._writer_track.release()
                self._writer_track = None

        # Open a writer for the incoming state.
        if self._state.storage is not None:
            self._open_storage_writer(new_state)
        else:
            self._open_legacy_writer(new_state)

    def _release_all_writers(self) -> None:
        """Release both video writers (idempotent)."""
        if self._writer_idle is not None:
            self._writer_idle.release()
            self._writer_idle = None
        if self._writer_track is not None:
            self._writer_track.release()
            self._writer_track = None
