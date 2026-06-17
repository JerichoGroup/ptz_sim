#!/usr/bin/env python3
"""
PTZ Sim terminal controller — run over SSH, no display needed.

Usage:
    # From the Jetson — point --host at the host PC running PTZSim/Isaac Sim:
    python3 ptz_tui.py --host <HOST_PC_IP>

    # On the host PC itself:
    python3 ptz_tui.py --host 127.0.0.1

    # 'm' toggles between the --host target ("remote") and 127.0.0.1 ("local").
    python3 ptz_tui.py --host <HOST_PC_IP> --host-port 5005 --listen-port 5006
"""

import argparse
import curses
import time

from ptz_sim_controller import PTZSimController

# ── Tunable defaults ─────────────────────────────────────────────────────────
MOVE_VEL   = 0.5    # normalised velocity sent by arrow keys
REL_DELTA  = 0.05   # delta for WASD relative_move
CAZ_ZOOM   = 0.70   # zoom target used by the 'c' center_and_zoom test

JETSON_IP  = "192.168.30.171"
HOST_IP    = "127.0.0.1"


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _safe_addstr(win, row, col, text, attr=0):
    h, w = win.getmaxyx()
    if row < 0 or row >= h or col < 0:
        return
    max_len = w - col - 1
    if max_len <= 0:
        return
    try:
        win.addstr(row, col, text[:max_len], attr)
    except curses.error:
        pass


def _draw(win, ptz, last_cmd, mode):
    win.erase()
    _, W = win.getmaxyx()
    sep = "─" * (W - 1)
    bold = curses.A_BOLD

    host_str = f"{ptz.host_addr[0]}:{ptz.host_addr[1]}"
    _safe_addstr(win, 0, 0,
        f" PTZ Sim  [mode: {mode}]  →  {host_str}    [q=quit  m=toggle mode]", bold)
    _safe_addstr(win, 1, 0, sep)

    pan, tilt, zoom, ts = ptz.get_pose()
    lag = time.time() - ts if ts else 0.0

    _safe_addstr(win, 2, 2, "Pose (from host):")
    _safe_addstr(win, 3, 4, f"Pan:  {pan:+9.2f}°    Tilt: {tilt:+9.2f}°    Zoom: {zoom:.3f}")
    _safe_addstr(win, 4, 4, f"vel_scale: {ptz.vel_scale():.3f}    "
                             f"rel_moving: {ptz._rel_moving!s:<5}    "
                             f"zooming: {ptz._zooming!s:<5}    "
                             f"lag: {lag*1000:.0f} ms")

    _safe_addstr(win, 5, 0, sep)
    _safe_addstr(win, 6, 2, "Controls:", bold)

    controls = [
        ("← → ↑ ↓",  "continuous pan/tilt  (hold key)"),
        ("+ -",       "zoom_in / zoom_out"),
        ("[ ]",       "zoom_search / zoom_track"),
        ("w a s d",   f"relative_move ±{REL_DELTA}  (tilt+ pan- tilt- pan+)"),
        ("c",         f"center_and_zoom(dp=0.02, dt=0.02, zoom={CAZ_ZOOM})"),
        ("h",         "save_home"),
        ("g",         "go_home"),
        ("SPACE",     "stop  (move 0, 0)"),
        ("m",         "toggle target  remote (--host) / local (127.0.0.1)"),
        ("q",         "quit"),
    ]
    for i, (key, desc) in enumerate(controls):
        _safe_addstr(win, 7 + i, 4, f"{key:<10}  {desc}")

    _safe_addstr(win, 7 + len(controls), 0, sep)
    _safe_addstr(win, 8 + len(controls), 2, f"Last: {last_cmd}")

    win.refresh()


# ── Main TUI loop ─────────────────────────────────────────────────────────────

def _tui(stdscr, ptz, remote_ip, host_port):
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    stdscr.timeout(50)   # 50 ms ≈ 20 Hz, well within cmd_ttl=0.15 s

    # Two targets: the remote host passed via --host, and local loopback.
    # 'm' toggles between them, so the same TUI works on the Jetson (remote)
    # or on the host PC itself (local).
    targets = {"remote": remote_ip, "local": HOST_IP}
    mode = "local" if ptz.host_addr[0] == HOST_IP else "remote"
    last_cmd = "—"

    while True:
        _draw(stdscr, ptz, last_cmd, mode)

        key = stdscr.getch()

        # ── Quit ─────────────────────────────────────────────────────────────
        if key == ord('q'):
            ptz.stop_all()
            break

        # ── Mode toggle ───────────────────────────────────────────────────────
        elif key == ord('m'):
            mode = "local" if mode == "remote" else "remote"
            new_ip = targets[mode]
            ptz.host_addr = (new_ip, host_port)
            last_cmd = f"target → {mode}  ({new_ip}:{host_port})"

        # ── Continuous pan/tilt (hold key → repeated getch → refreshed TTL) ─
        elif key == curses.KEY_LEFT:
            ptz.move(-MOVE_VEL, 0.0)
            last_cmd = f"move({-MOVE_VEL}, 0.0)  [pan left]"

        elif key == curses.KEY_RIGHT:
            ptz.move(MOVE_VEL, 0.0)
            last_cmd = f"move({MOVE_VEL}, 0.0)  [pan right]"

        elif key == curses.KEY_UP:
            ptz.move(0.0, MOVE_VEL)
            last_cmd = f"move(0.0, {MOVE_VEL})  [tilt up]"

        elif key == curses.KEY_DOWN:
            ptz.move(0.0, -MOVE_VEL)
            last_cmd = f"move(0.0, {-MOVE_VEL})  [tilt down]"

        # ── Zoom ─────────────────────────────────────────────────────────────
        elif key in (ord('+'), ord('=')):
            ptz.zoom_in()
            last_cmd = "zoom_in()"

        elif key == ord('-'):
            ptz.zoom_out()
            last_cmd = "zoom_out()"

        elif key == ord('['):
            ptz.zoom_search()
            last_cmd = "zoom_search()"

        elif key == ord(']'):
            ptz.zoom_track()
            last_cmd = "zoom_track()"

        # ── Relative move (WASD) ─────────────────────────────────────────────
        elif key == ord('w'):
            ptz.relative_move(0.0, REL_DELTA)
            last_cmd = f"relative_move(dp=0, dt=+{REL_DELTA})  [tilt up]"

        elif key == ord('s'):
            ptz.relative_move(0.0, -REL_DELTA)
            last_cmd = f"relative_move(dp=0, dt={-REL_DELTA})  [tilt down]"

        elif key == ord('a'):
            ptz.relative_move(-REL_DELTA, 0.0)
            last_cmd = f"relative_move(dp={-REL_DELTA}, dt=0)  [pan left]"

        elif key == ord('d'):
            ptz.relative_move(REL_DELTA, 0.0)
            last_cmd = f"relative_move(dp=+{REL_DELTA}, dt=0)  [pan right]"

        # ── Center and zoom ───────────────────────────────────────────────────
        elif key == ord('c'):
            fired = ptz.center_and_zoom(0.02, 0.02, CAZ_ZOOM)
            last_cmd = f"center_and_zoom(0.02, 0.02, {CAZ_ZOOM}) → {'fired' if fired else 'blocked (busy)'}"

        # ── Home ─────────────────────────────────────────────────────────────
        elif key == ord('h'):
            ptz.save_home()
            last_cmd = "save_home()"

        elif key == ord('g'):
            ptz.go_home()
            last_cmd = "go_home()"

        # ── Force stop ───────────────────────────────────────────────────────
        elif key == ord(' '):
            ptz.move(0.0, 0.0)
            last_cmd = "move(0.0, 0.0)  [stop]"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PTZ Sim terminal controller")
    parser.add_argument("--host",        default=JETSON_IP, help="Host running PTZSim")
    parser.add_argument("--host-port",   type=int, default=5005)
    parser.add_argument("--listen-port", type=int, default=5006)
    args = parser.parse_args()

    print(f"Connecting to PTZSim at {args.host}:{args.host_port} ...")

    ptz = PTZSimController(
        host_ip=args.host,
        host_port=args.host_port,
        listen_port=args.listen_port,
    )

    time.sleep(0.3)
    curses.wrapper(_tui, ptz, args.host, args.host_port)
    print("TUI closed.")


if __name__ == "__main__":
    main()
