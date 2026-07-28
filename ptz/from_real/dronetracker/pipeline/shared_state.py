"""Thread-safe shared state for the live PTZ pipeline.

Replaces the 13 single-element list "mailbox" globals in the original
live PTZ script with a single ``SharedState`` object whose named attributes
map 1:1 to the original globals.

Thread ownership:
  ``frame_lock``   — grab thread writes ``latest_frame``; detect + main read it.
  ``det_lock``     — detect thread writes all ``det_*`` fields; main reads them.
  ``cmd_lock``     — main thread writes ``cmd_lock_on`` / ``cmd_unlock``; detect reads them.
  ``recording``, ``video_writer`` — main thread only; not guarded by a lock.
  ``stop_ev``      — written by main (set), read by grab + detect threads.
"""

__all__ = ["SharedState"]

import threading


class SharedState:
    """Container for all cross-thread state in the live PTZ pipeline.

    Attribute groups mirror the original list-cell globals; lock semantics
    are preserved exactly so the detect / grab / main threads interact in
    the same way as before.
    """

    # ── Frame pipeline ────────────────────────────────────────────────────────
    def __init__(self):
        self.frame_lock   = threading.Lock()
        self.stop_ev      = threading.Event()
        self.latest_frame = None          # guarded by frame_lock

        # ── Detect → Main (read under det_lock) ──────────────────────────────
        self.det_lock         = threading.Lock()
        self.state            = "IDLE"
        self.tracks           = []         # list of Norfair TrackedObject
        self.fps              = 0.0
        self.locked_id        = None       # Norfair track ID currently locked
        self.yolo_box         = None       # (x1,y1,x2,y2,conf,label) or None
        self.yolo_boxes       = []         # list of (x1,y1,x2,y2,conf,label) for all YOLO dets; [] when none
        self.yolo_box_followed = False     # True if yolo_box is the actively-followed target (red), else raw/fallback (cyan)
        self.follow_aim       = None       # (px,py) camera-px predicted aim point for crosshair, or None
        self.request_shot     = False      # detect → main: take screenshot now

        # ── Main only (no lock needed) ────────────────────────────────────────
        self.recording        = False
        self.video_writer     = None

        # Segmented SSD storage (StorageManager or None when storage is disabled).
        # Set once by the pipeline orchestrator before the threads start; the
        # display thread drives rotation/quota and the detect thread appends
        # JSON/crops.  The StorageManager carries its own internal lock.
        self.storage          = None

        # ── Main → Detect (written by main, consumed by detect once per frame) ─
        self.cmd_lock         = threading.Lock()
        self.cmd_lock_on      = False    # IDLE → TRACK transition request
        self.cmd_unlock       = False    # any state → return-home request

    # ── Frame accessors ───────────────────────────────────────────────────────

    def get_frame(self):
        """Return the latest raw frame (or None if none received yet)."""
        with self.frame_lock:
            return self.latest_frame

    def set_frame(self, frame) -> None:
        """Store a new frame from the grab thread."""
        with self.frame_lock:
            self.latest_frame = frame

    # ── Detect-side publisher ─────────────────────────────────────────────────

    def publish_det(
        self,
        state        = None,
        tracks       = None,
        fps          = None,
        locked_id    = None,
        yolo_box     = None,
        request_shot = None,
    ) -> None:
        """Atomically update any subset of the detect-side fields under det_lock.

        Only the keyword arguments that are not ``None`` are updated.  Callers
        can publish a single field without touching the others::

            state.publish_det(state="IDLE", locked_id=None)
        """
        with self.det_lock:
            if state        is not None: self.state        = state
            if tracks       is not None: self.tracks       = tracks
            if fps          is not None: self.fps          = fps
            if locked_id    is not None: self.locked_id    = locked_id
            if yolo_box     is not None: self.yolo_box     = yolo_box
            if request_shot is not None: self.request_shot = request_shot

    def clear_locked_id(self) -> None:
        """Set locked_id to None under det_lock (common operation)."""
        with self.det_lock:
            self.locked_id = None

    def set_yolo_boxes(self, dets: list) -> None:
        """Atomically store the full list of YOLO detections for display."""
        with self.det_lock:
            self.yolo_boxes = list(dets)

    def clear_yolo_all(self) -> None:
        """Atomically clear yolo_box, yolo_boxes, and the follow overlay state."""
        with self.det_lock:
            self.yolo_box          = None
            self.yolo_boxes        = []
            self.yolo_box_followed = False
            self.follow_aim        = None

    def set_track_overlay(self, yolo_box, followed: bool, aim) -> None:
        """Atomically set the TRACK overlay in one lock acquisition.

        Args:
            yolo_box: the single box to draw (x1,y1,x2,y2,conf,label), or None.
            followed: True if yolo_box is the actively-followed target (drawn red
                      'FOLLOW'); False if it is a raw/fallback detection (cyan 'YOLO').
            aim:      (px, py) camera-px predicted aim point for the crosshair, or
                      None.  Shown even when yolo_box is None (follower coasting).
        """
        with self.det_lock:
            self.yolo_box          = yolo_box
            self.yolo_box_followed = followed
            self.follow_aim        = aim

    # ── Detect-side snapshot (for main thread display) ─────────────────────────

    def snapshot_for_display(self) -> dict:
        """Return a shallow copy of all det-side fields in a single lock acquisition.

        The caller must not hold det_lock when calling this.
        """
        with self.det_lock:
            return {
                "state":         self.state,
                "tracks":        list(self.tracks),
                "fps":           self.fps,
                "locked_id":     self.locked_id,
                "yolo_box":      self.yolo_box,
                "yolo_boxes":    list(self.yolo_boxes),
                "yolo_box_followed": self.yolo_box_followed,
                "follow_aim":    self.follow_aim,
                "request_shot":  self.request_shot,
            }

    def take_shot_request(self) -> bool:
        """Atomically read-and-clear ``request_shot``; returns previous value."""
        with self.det_lock:
            val = self.request_shot
            self.request_shot = False
        return val

    # ── Command accessors (main → detect) ─────────────────────────────────────

    def post_lock_command(self) -> None:
        """Signal the detect thread to transition IDLE → TRACK (T/Enter key)."""
        with self.cmd_lock:
            self.cmd_lock_on = True

    def post_unlock_command(self) -> None:
        """Signal the detect thread to return home from any state (R key)."""
        with self.cmd_lock:
            self.cmd_unlock = True

    def drain_commands(self) -> tuple:
        """Atomically consume both command flags; returns ``(do_lock, do_unlock)``."""
        with self.cmd_lock:
            do_lock   = self.cmd_lock_on;  self.cmd_lock_on  = False
            do_unlock = self.cmd_unlock;   self.cmd_unlock   = False
        return do_lock, do_unlock
