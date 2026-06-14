#!/usr/bin/env python3
import time
from ptz_sim_controller import PTZSimController


def main():
    # CHANGE THIS to the machine running your UDP listener
    HOST_IP = "192.168.30.171"   # your machine
    HOST_PORT = 5005             # must match listener
    LISTEN_PORT = 5006           # for pose updates (optional)

    print(f"[Jetson] Starting PTZSimController → {HOST_IP}:{HOST_PORT}")

    ptz = PTZSimController(
        host_ip=HOST_IP,
        host_port=HOST_PORT,
        listen_port=LISTEN_PORT,
    )

    time.sleep(1)
    print("[Jetson] Controller ready, sending test commands...\n")

    # --- Test sequence ----------------------------------------------------
    while True:
        print("[Jetson] move(0.2, 0.0)")
        ptz.move(0.2, 0.0)
        time.sleep(1)

        print("[Jetson] move(0.0, -0.2)")
        ptz.move(0.0, -0.2)
        time.sleep(1)

        print("[Jetson] relative_move(0.1, 0.05)")
        ptz.relative_move(0.1, 0.05)
        time.sleep(1)

        print("[Jetson] zoom_in()")
        ptz.zoom_in()
        time.sleep(1)

        print("[Jetson] zoom_out()")
        ptz.zoom_out()
        time.sleep(1)

        print("[Jetson] center_and_zoom(0.05, -0.05, 0.7)")
        ptz.center_and_zoom(0.05, -0.05, 0.7)
        time.sleep(2)

        print("[Jetson] save_home()")
        ptz.save_home()
        time.sleep(1)

        print("[Jetson] go_home()")
        ptz.go_home()
        time.sleep(2)

        print("[Jetson] Looping...\n")
