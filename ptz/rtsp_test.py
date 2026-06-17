#!/usr/bin/env python3
"""Low-latency RTSP stream viewer for diagnosing glass-to-glass delay.

Two backends are available so we can isolate WHERE latency comes from:

  --backend gst     (default) OpenCV via GStreamer. rtspsrc latency=0 +
                    appsink sync=false drop=true → near-zero client buffering.
                    This is the true low-latency path.

  --backend ffmpeg  OpenCV via FFMPEG — the SAME backend the Jetson's
                    RtspGrabber uses. FFMPEG's RTSP demuxer keeps a stubborn
                    internal buffer (~1-2 s) that survives max_delay tweaks.

If 'gst' is low-latency but 'ffmpeg' is ~2 s, the bottleneck is the FFMPEG
client decoder, not our server.

Usage:
    python3 ptz/rtsp_test.py
    python3 ptz/rtsp_test.py --backend ffmpeg
    python3 ptz/rtsp_test.py --url rtsp://192.168.30.171:8554/unicast/c1/s0/live
"""

import argparse
import os
import time
import cv2

DEFAULT_URL = "rtsp://127.0.0.1:8554/unicast/c1/s0/live"


def open_gst(url):
    # latency=0          : no rtspsrc jitter buffer
    # drop-on-latency    : discard late packets instead of waiting
    # appsink sync=false : display frames as soon as decoded, no clock sync wait
    # drop=true max-buffers=1 : only ever hold the newest decoded frame
    pipeline = (
        f"rtspsrc location={url} latency=0 drop-on-latency=true protocols=udp "
        "! rtph264depay ! h264parse ! avdec_h264 "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink sync=false drop=true max-buffers=1"
    )
    print(f"[gst] {pipeline}")
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def open_ffmpeg(url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|"
        "max_delay;0|reorder_queue_size;0|stimeout;5000000"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--backend", choices=["gst", "ffmpeg"], default="gst")
    args = parser.parse_args()

    print(f"Connecting to {args.url}  (backend={args.backend}) ...")
    cap = open_gst(args.url) if args.backend == "gst" else open_ffmpeg(args.url)

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
