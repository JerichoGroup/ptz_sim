#!/usr/bin/env python3
"""
PTZ Sim terminal controller — run from Jetson over SSH, no display needed.

Usage:
    python3 ptz_tui.py --host <host-ip>
    python3 ptz_tui.py --host 192.168.55.2 --host-port 5005 --listen-port 5006
"""

import argparse
import curses
import time

from ptz_sim_controller import PTZSimController

# ── Tunable defaults ─────────────────────────────────────────────────────────
MOVE_VEL   = 0.5    # normalised velocity sent by arrow keys
REL_DELTA  = 0.05   # delta for WASD relative_move
CAZ_ZOOM   = 0.70   # zoom target used by the 'c' center_and_zoom test


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _safe_addstr(win, row, col, text, attr=0):
    """addstr that silently drops the call if it would overflow the window."""
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


def _draw(win, ptz, last_cmd, host_addr):
    win.erase()
    _, W = win.getmaxyx()
    sep = "─" * (W - 1)

    bold = curses.A_BOLD

    _safe_addstr(win, 0, 0, f" PTZ Sim Controller  →  {host_addr}    [q = quit]", bold)
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
        ("← → ↑ ↓",    "continuous pan/tilt  (hold key)"),
        ("+ -",         "zoom_in / zoom_out"),
        ("[ ]",         "zoom_search / zoom_track"),
        ("w a s d",     f"relative_move ±{REL_DELTA}  (tilt+ pan- tilt- pan+)"),
        ("c",           f"center_and_zoom(dp=0.02, dt=0.02, zoom={CAZ_ZOOM})"),
        ("h",           "save_home"),
        ("g",           "go_home"),
        ("SPACE",       "stop  (move 0, 0)"),
        ("q",           "quit"),
    ]
    for i, (key, desc) in enumerate(controls):
        _safe_addstr(win, 7 + i, 4, f"{key:<10}  {desc}")

    _safe_addstr(win, 7 + len(controls), 0, sep)
    _safe_addstr(win, 8 + len(controls), 2, f"Last: {last_cmd}")

    win.refresh()


# ── Main TUI loop ─────────────────────────────────────────────────────────────

def _tui(stdscr, ptz, host_addr):
    curses.curs_set(0)
    stdscr.keypad(True)     # enable KEY_UP, KEY_LEFT, etc.
    stdscr.nodelay(True)    # non-blocking getch
    stdscr.timeout(50)      # 50 ms ≈ 20 Hz — within the host's cmd_ttl=0.15 s

    last_cmd = "—"

    while True:
        _draw(stdscr, ptz, last_cmd, host_addr)

        key = stdscr.getch()

        # ── Quit ─────────────────────────────────────────────────────────────
        if key == ord('q'):
            ptz.stop_all()
            break

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

        # When no arrow key is held, the host's cmd_ttl=0.15 s naturally expires
        # and the mover loop stops — no explicit stop packet needed.


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PTZ Sim terminal controller (Jetson-side)")
    parser.add_argument("--host",        default="192.168.30.171", help="IP of the host running PTZSim")
    parser.add_argument("--host-port",   type=int, default=5005)
    parser.add_argument("--listen-port", type=int, default=5006)
    args = parser.parse_args()

    host_addr = f"{args.host}:{args.host_port}"
    print(f"Connecting to PTZSim at {host_addr} ...")

    ptz = PTZSimController(
        host_ip=args.host,
        host_port=args.host_port,
        listen_port=args.listen_port,
    )

    time.sleep(0.3)   # give the pose listener socket time to bind
    curses.wrapper(_tui, ptz, host_addr)
    print("TUI closed.")


if __name__ == "__main__":
    main()
