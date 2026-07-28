"""RTSP frame grabber for the live PTZ pipeline.

``RtspGrabber`` runs as a daemon thread, continuously reading frames from the
camera and writing them to ``SharedState.latest_frame`` under ``frame_lock``.
It alternates between two RTSP URLs on failure and applies a garbled-frame
filter before storing each frame.
"""

__all__ = ["RtspGrabber", "is_garbled"]

import os
import time
import threading


def is_garbled(frame, prev_frame, threshold: float = 60.0) -> bool:
    """Return True if ``frame`` is a likely corrupt/garbled frame.

    Detects large abrupt pixel changes by computing the mean absolute
    difference of downscaled versions.  A difference above ``threshold``
    (empirically: >60 on a 160×90 thumbnail) indicates a codec artefact
    rather than scene motion.

    Args:
        frame:      Current frame (BGR numpy array).
        prev_frame: Previous accepted frame (same shape).
        threshold:  Mean-pixel-diff threshold above which the frame is flagged.
    """
    import cv2
    d = float(cv2.absdiff(
        cv2.resize(frame,      (160, 90)),
        cv2.resize(prev_frame, (160, 90)),
    ).mean())
    return d > threshold


class RtspGrabber:
    """Daemon thread that fills ``SharedState.latest_frame`` from an RTSP stream.

    Alternates between ``url_main`` and ``url_alt`` on connection failure.
    Drops garbled frames (large abrupt pixel diff vs. the previous good frame).
    """

    def __init__(self, url_main: str, url_alt: str, shared_state):
        """
        Args:
            url_main:     Primary RTSP URL.
            url_alt:      Fallback RTSP URL (tried on cap.read() failure).
            shared_state: ``SharedState`` instance; ``set_frame()`` is called
                          for each accepted frame; ``stop_ev`` signals shutdown.
        """
        self._urls   = [url_main, url_alt]
        self._state  = shared_state
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start the grabber thread (non-blocking)."""
        self._thread.start()

    def _run(self) -> None:
        """Main grab loop — runs until ``shared_state.stop_ev`` is set."""
        import cv2

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|strict;-2"
        )
        url_idx = 0
        stop_ev = self._state.stop_ev

        while not stop_ev.is_set():
            url = self._urls[url_idx % 2]
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"[Cam] Cannot open {url}")
                cap.release()
                url_idx += 1
                time.sleep(2)
                continue

            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Cam] {fw}x{fh} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")

            last_good = None
            consecutive_dropped = 0
            while not stop_ev.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    break
                if last_good is not None and is_garbled(frame, last_good):
                    consecutive_dropped += 1
                    if consecutive_dropped < 5:
                        continue
                    # 5+ consecutive drops → real scene change, not a codec
                    # artifact.  Accept this frame as the new baseline.
                consecutive_dropped = 0
                last_good = frame
                self._state.set_frame(frame)

            cap.release()
            url_idx += 1
