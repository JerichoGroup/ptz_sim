"""Offline video processing runner.

``OfflineRunner`` implements the open->seek->init->read->detect->track->draw->write
loop shared by all offline scripts.  The detection logic is injected as a
callable, keeping the runner generic.

Typical use::

    def detect_fn(frame):
        mask = mt.detect_motion(frame)
        return motion_to_detections(mask, ...)

    runner = OfflineRunner(detect_fn=detect_fn, tracker=tracker)
    runner.run(video_path, output_path, start_sec=270.0)

When a ``track_filter_cfg`` is provided and ``enabled=True``, a streaming
``TrackFilter`` is applied every frame after ``tracker.update()``.  Tracks
that fail the current criteria are removed from the draw list that frame
(stop drawing).  This is identical to how the live PTZ pipeline filters
tracks — no second pass, no retroactive removal.

When ``track_filter_cfg`` is None or ``enabled=False``, every confirmed track
is drawn (no-op).
"""

__all__ = ["OfflineRunner"]

import cv2
import numpy as np
from tqdm import tqdm

from dronetracker.media.video import make_writer, AVI_FOURCC
from dronetracker.tracking.trails import track_history, clear_trails
from dronetracker.tracking.filtering import TrackFilter


def _default_draw(frame, tracked_objects):
    """Minimal track overlay: green dot + ID label + blue trail.

    The track the auto-hunt evaluator would zoom on (``obj.auto_engage``) is
    drawn in magenta with a ``ZOOM`` tag.
    """
    for obj in tracked_objects:
        if obj.last_detection is None:
            continue
        track_id = obj.id
        x, y = int(obj.estimate[0][0]), int(obj.estimate[0][1])
        zoom  = getattr(obj, "auto_engage", False)
        color = (255, 0, 255) if zoom else (0, 255, 0)   # magenta = would-zoom
        cv2.circle(frame, (x, y), 9 if zoom else 5, color, -1)
        score = getattr(obj, "drone_score", None)
        label = f"ID {track_id}" if score is None else f"ID {track_id} {score:.2f}"
        if zoom:
            label = "ZOOM " + label
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        trail = track_history[track_id]
        trail.append((x, y))
        if len(trail) > 50:
            trail.pop(0)
        if len(trail) > 1:
            trail_color = (255, 0, 255) if zoom else (255, 0, 0)  # magenta = would-zoom
            cv2.polylines(frame, [np.array(trail, dtype=np.int32)],
                          False, trail_color, 2)
    return frame


class OfflineRunner:
    """Offline video processing pipeline with injectable detection.

    The runner manages the video I/O loop; all detection logic lives in
    ``detect_fn`` so the same runner can host motion-diff or tiled-YOLO.

    Args:
        detect_fn:        ``(frame) -> list[Detection]``.  Called for every frame
                          after the first.  Must be stateful if periodic side-effects
                          (e.g. ground-mask recompute) are needed -- use a closure.
        tracker:          A Norfair ``Tracker`` instance (or equivalent with
                          ``tracker.update(detections=...) -> list``).
        init_fn:          Optional ``(first_frame, fps, width, height) -> None``.
                          Called once after the video is opened and before the main
                          loop, with the first decoded frame.  Use it to initialise
                          detectors that need the frame size or the first image
                          (e.g. ``GroundMaskGenerator.compute``).
        draw_fn:          ``(frame, tracked_objects) -> frame``.  Defaults to
                          a simple dot + trail overlay.  Only called with the
                          post-filter track list, so dropped tracks are never drawn.
        track_filter_cfg: Optional ``TrackFilterConfig``.  When set and
                          ``enabled=True``, applies the streaming whole-life filter
                          every frame — identical to the live pipeline.
        evaluator:        Optional ``AutoEngageEvaluator``.  Run on the surviving
                          (post-filter) tracks each frame to stamp
                          ``obj.drone_score`` (rendered by the draw fn) and log
                          feature vectors.  Offline has no PTZ, so its return value
                          (the engage candidate) is ignored — score + annotate + log
                          only.
        fourcc:           VideoWriter codec FourCC.  Defaults to ``AVI_FOURCC``
                          (XVID), which produces files that are readable mid-write.
        show_window:      Show a live ``cv2.imshow`` preview window.  Leave False
                          for headless / background runs.
    """

    def __init__(
        self,
        detect_fn,
        tracker,
        init_fn=None,
        draw_fn=None,
        track_filter_cfg=None,
        evaluator=None,
        fourcc: int = AVI_FOURCC,
        show_window: bool = False,
        after_update=None,
        on_video_start=None,
        augment_fn=None,
        observe_fn=None,
    ):
        self._detect_fn        = detect_fn
        self._tracker          = tracker
        self._init_fn          = init_fn
        self._draw_fn          = draw_fn if draw_fn is not None else _default_draw
        self._track_filter_cfg = track_filter_cfg
        self._evaluator        = evaluator
        self._fourcc           = fourcc
        self._show_window      = show_window
        # Optional ``(tracked_objects) -> None`` called right after every
        # tracker.update() (full list, before filtering) — e.g. to update a
        # per-track appearance template store.  Runs on this single loop thread.
        self._after_update     = after_update
        # Optional ``() -> None`` called once per video at start, next to
        # clear_trails() — e.g. to reset per-track state between batch videos.
        self._on_video_start   = on_video_start
        # Optional ``(frame, prev_tracked, motion_dets) -> (extra_dets, drop_idxs)``
        # called BEFORE tracker.update() to inject synthetic detections (e.g. the
        # template tracker pinning a slow object) and drop suppressed motion dets.
        self._augment_fn       = augment_fn
        # Optional ``(frame, tracked_objects) -> None`` after tracker.update() —
        # like after_update but also receives the frame (e.g. to refresh template
        # patches from the current image).
        self._observe_fn       = observe_fn

    def run(
        self,
        video_path: str,
        output_path: str,
        start_sec: float = 0.0,
        end_sec: float = None,
        window_name: str = "Offline Tracker",
    ) -> None:
        """Run the full offline pipeline to completion.

        Args:
            video_path:  Input video file path.
            output_path: Output annotated video path.
            start_sec:   Seek to this timestamp (seconds) before processing.
            end_sec:     Stop processing at this timestamp (seconds). None = run to EOF.
            window_name: cv2 window title (only used if ``show_window=True``).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps          = int(cap.get(cv2.CAP_PROP_FPS))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Build the streaming filter now that frame dims are known.
        cfg = self._track_filter_cfg
        track_filter = (
            TrackFilter(width, height, cfg, fps=max(1, fps))
            if cfg is not None and getattr(cfg, "enabled", False)
            else None
        )

        writer = make_writer(output_path, fps, width, height, self._fourcc)

        if self._show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

        if self._init_fn is not None:
            ret, first_frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read first frame from: {video_path}")
            self._init_fn(first_frame, fps, width, height)

        seen_ids  = set()
        drawn_ids = set()
        if end_sec is not None:
            remaining = int((end_sec - start_sec) * fps)
        else:
            remaining = total_frames - int(start_sec * fps)

        # Clear any trail history left by a previous run (critical in batch
        # mode: each video gets a fresh Norfair Tracker whose IDs restart at 0,
        # so without this a track in video N inherits the trail of the same ID
        # from video N-1).
        clear_trails()
        if self._on_video_start is not None:
            self._on_video_start()

        end_ms = end_sec * 1000 if end_sec is not None else None
        prev_tracked = []

        fps_safe  = max(fps, 1)   # video-time clock for the evaluator's history window
        frame_no  = 0

        with tqdm(total=remaining) as pbar:
            while True:
                if end_ms is not None and cap.get(cv2.CAP_PROP_POS_MSEC) >= end_ms:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self._detect_fn(frame)

                # Inject synthetic detections (e.g. template-pinned slow objects)
                # and drop the motion detections they suppress.
                if self._augment_fn is not None:
                    extra, drop = self._augment_fn(frame, prev_tracked, detections)
                    if drop:
                        detections = [d for i, d in enumerate(detections) if i not in drop]
                    detections = detections + list(extra)

                tracked_objects = self._tracker.update(detections=detections)
                prev_tracked = tracked_objects

                if self._after_update is not None:
                    self._after_update(tracked_objects)
                if self._observe_fn is not None:
                    self._observe_fn(frame, tracked_objects)

                if track_filter is not None:
                    # Accumulate all confirmed IDs before filtering.
                    seen_ids.update(
                        obj.id for obj in tracked_objects
                        if obj.last_detection is not None
                    )
                    survivors       = track_filter.update(tracked_objects)
                    tracked_objects = [o for o in tracked_objects
                                       if o.id in survivors]
                    drawn_ids.update(
                        obj.id for obj in tracked_objects
                        if obj.last_detection is not None
                    )

                # Score the surviving tracks (stamps obj.drone_score + logs).
                # No PTZ offline, so the returned engage candidate is ignored.
                if self._evaluator is not None:
                    self._evaluator.update(tracked_objects, None, frame_no / fps_safe)

                frame = self._draw_fn(frame, tracked_objects)
                writer.write(frame)
                frame_no += 1

                if self._show_window:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) == 27:
                        break

                pbar.update(1)

        cap.release()
        writer.release()
        if self._evaluator is not None:
            self._evaluator.close()
        if self._show_window:
            cv2.destroyAllWindows()

        if track_filter is not None:
            total_seen    = len(seen_ids)
            total_drawn   = len(drawn_ids)
            total_dropped = total_seen - total_drawn
            print(f"Filter: {total_drawn} drawn, {total_dropped} dropped "
                  f"(of {total_seen} total tracks)")
        print("Done.")
