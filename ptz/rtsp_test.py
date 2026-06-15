#!/usr/bin/env python3
"""Quick RTSP stream viewer — validates the same code path as the Jetson's RtspGrabber.

Usage:
    python3 ptz/rtsp_test.py
    python3 ptz/rtsp_test.py --url rtsp://192.168.30.171:8554/unicast/c1/s0/live
"""

import argparse
import time
import cv2

DEFAULT_URL = "rtsp://127.0.0.1:8554/unicast/c1/s0/live"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    # Match the exact OpenCV flags RtspGrabber uses
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|strict;-2"
    )

    print(f"Connecting to {args.url} ...")
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: could not open RTSP stream")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Connected: {w}x{h}")

    cv2.namedWindow("RTSP test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RTSP test", 960, 540)

    t0 = time.time()
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or lost — retrying...")
            time.sleep(0.5)
            continue

        n += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            print(f"FPS: {n / elapsed:.1f}")
            n = 0
            t0 = time.time()

        cv2.imshow("RTSP test", frame)
        if cv2.waitKey(1) == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
