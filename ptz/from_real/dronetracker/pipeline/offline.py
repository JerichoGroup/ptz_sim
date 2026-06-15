"""Offline video processing runner.

``OfflineRunner`` implements the open→seek→init→read→detect→track→draw→write
loop shared by all offline scripts.  The detection logic is injected as a
callable, keeping the runner generic.

Typical use::

    def detect_fn(frame):
        mask = mt.detect_motion(frame)
        return motion_to_detections(mask, ...)

    runner = OfflineRunner(detect_fn=detect_fn, tracker=tracker)
    runner.run(video_path, output_path, start_sec=270.0)
"""

__all__ = ["OfflineRunner"]

import cv2
import numpy as np
from tqdm import tqdm

from dronetracker.media.video import make_writer, AVI_FOURCC
from dronetracker.tracking.trails import track_history


def _default_draw(frame, tracked_objects):
    """Minimal track overlay: green dot + ID label + blue trail."""
    for obj in tracked_objects:
        if obj.last_detection is None:
            continue
        track_id = obj.id
        x, y = int(obj.estimate[0][0]), int(obj.estimate[0][1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {track_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        trail = track_history[track_id]
        trail.append((x, y))
        if len(trail) > 50:
            trail.pop(0)
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, dtype=np.int32)],
                          False, (255, 0, 0), 2)
    return frame


class OfflineRunner:
    """Offline video processing pipeline with injectable detection.

    The runner manages the video I/O loop; all detection logic lives in
    ``detect_fn`` so the same runner can host motion-diff or tiled-YOLO.

    Args:
        detect_fn:    ``(frame) -> list[Detection]``.  Called for every frame
                      after the first.  Must be stateful if periodic side-effects
                      (e.g. ground-mask recompute) are needed — use a closure.
        tracker:      A Norfair ``Tracker`` instance (or equivalent with
                      ``tracker.update(detections=...) -> list``).
        init_fn:      Optional ``(first_frame, fps, width, height) -> None``.
                      Called once after the video is opened and before the main
                      loop, with the first decoded frame.  Use it to initialise
                      detectors that need the frame size or the first image
                      (e.g. ``GroundMaskGenerator.compute``).
        draw_fn:      ``(frame, tracked_objects) -> frame``.  Defaults to
                      a simple dot + trail overlay.
        fourcc:       VideoWriter codec FourCC.  Defaults to ``AVI_FOURCC``
                      (XVID), which produces files that are readable mid-write.
        show_window:  Show a live ``cv2.imshow`` preview window.  Leave False
                      for headless / background runs.
    """

    def __init__(
        self,
        detect_fn,
        tracker,
        init_fn=None,
        draw_fn=None,
        fourcc: int = AVI_FOURCC,
        show_window: bool = False,
    ):
        self._detect_fn   = detect_fn
        self._tracker     = tracker
        self._init_fn     = init_fn
        self._draw_fn     = draw_fn if draw_fn is not None else _default_draw
        self._fourcc      = fourcc
        self._show_window = show_window

    def run(
        self,
        video_path: str,
        output_path: str,
        start_sec: float = 0.0,
        window_name: str = "Offline Tracker",
    ) -> None:
        """Run the full offline pipeline to completion.

        Args:
            video_path:  Input video file path.
            output_path: Output annotated video path.
            start_sec:   Seek to this timestamp (seconds) before processing.
            window_name: cv2 window title (only used if ``show_window=True``).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps          = int(cap.get(cv2.CAP_PROP_FPS))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = make_writer(output_path, fps, width, height, self._fourcc)

        if self._show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

        # Read the first frame only when an init_fn needs it.
        # Without init_fn, skip this read so frame 0 is processed in the loop.
        if self._init_fn is not None:
            ret, first_frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read first frame from: {video_path}")
            self._init_fn(first_frame, fps, width, height)

        remaining = total_frames - int(start_sec * fps)
        with tqdm(total=remaining) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections      = self._detect_fn(frame)
                tracked_objects = self._tracker.update(detections=detections)
                frame           = self._draw_fn(frame, tracked_objects)
                writer.write(frame)

                if self._show_window:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) == 27:
                        break

                pbar.update(1)

        cap.release()
        writer.release()
        if self._show_window:
            cv2.destroyAllWindows()
        print("Done.")
