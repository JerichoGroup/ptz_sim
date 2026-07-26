"""Scene library for the PTZ simulator.

A *scene* flies a simulated target (a ``UdpBot`` drone) through a trajectory so
the Jetson's tracker has something to detect and lock onto. The Jetson selects a
scene by number via its config (``sim.scene``); that number is carried over the
existing PTZ command channel, and ``ptz_sim`` plays the scene a configurable
delay after Isaac Sim finishes loading.

Scenes are defined declaratively in ``scenes.yaml`` (sitting next to this file) —
add or edit a numbered scene there, no Python needed. Each scene is a list of
``moves`` driven against a fresh ``UdpBot``; ``run_scene`` creates, starts, and
closes the bot for you. The camera look-point is documented at the top of the
YAML so you can build trajectories around it.

If ``scenes.yaml`` is missing or PyYAML is unavailable, the built-in
``_FALLBACK_SCENES`` below are used instead.
"""

import os
import time

from isaac_core_dev_kit.udp.udp_bot import UdpBot


# UDP port Isaac Sim listens on for the target/drone prim (matches nirchuk.py).
DRONE_UDP_PORT = 33335

# Default target spawn pose (lat, lon, alt + orientation in degrees) — the camera
# look-point. Overridable via the ``start:`` block in scenes.yaml.
_DEFAULT_START = dict(
    start_lat=32.20647, start_lon=35.29034, start_alt=540.0,
    start_roll_d=0.0, start_pitch_d=0.0, start_yaw_d=0.0,
)

# Base time unit (seconds) used to pace the canned moves.
TIME_FOR_FIFTY = 2.0

# Path to the declarative scene definitions.
_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes.yaml")


# --- Move dispatch: maps a YAML move ``type`` to a UdpBot call -----------------

def _apply_move(bot: UdpBot, move: dict) -> None:
    """Execute a single declarative move (one entry of a scene's ``moves`` list)."""
    mtype = move.get("type")
    if mtype == "right_left":
        bot.move_right_left(distance_m=float(move["distance_m"]),
                            duration_s=float(move["duration_s"]))
    elif mtype == "forward_backward":
        bot.move_forward_backward(distance_m=float(move["distance_m"]),
                                  duration_s=float(move["duration_s"]))
    elif mtype == "up_down":
        bot.move_up_down(distance_m=float(move["distance_m"]),
                        duration_s=float(move["duration_s"]))
    elif mtype == "to_point":
        bot.move_to_point(
            target_lat=float(move["lat"]),
            target_lon=float(move["lon"]),
            target_alt=float(move["alt"]),
            target_yaw_d=float(move.get("yaw_d", 0.0)),
            target_roll_d=float(move.get("roll_d", 0.0)),
            target_pitch_d=float(move.get("pitch_d", 0.0)),
            duration_s=float(move["duration_s"]),
        )
    elif mtype == "wait":
        time.sleep(float(move["duration_s"]))
    else:
        print(f"[Scenes] unknown move type {mtype!r}; skipping")


# --- YAML loading --------------------------------------------------------------

def _load_yaml():
    """Load ``scenes.yaml``. Returns ``(start_kwargs, scenes_by_num)`` or ``None``.

    ``scenes_by_num`` maps int scene number -> list of move dicts.
    """
    try:
        import yaml
    except ImportError:
        print("[Scenes] PyYAML not available; using built-in scenes")
        return None
    if not os.path.exists(_YAML_PATH):
        print(f"[Scenes] {_YAML_PATH} not found; using built-in scenes")
        return None

    with open(_YAML_PATH) as f:
        data = yaml.safe_load(f) or {}

    s = data.get("start") or {}
    start_kwargs = dict(
        start_lat=float(s.get("lat", _DEFAULT_START["start_lat"])),
        start_lon=float(s.get("lon", _DEFAULT_START["start_lon"])),
        start_alt=float(s.get("alt", _DEFAULT_START["start_alt"])),
        start_roll_d=float(s.get("roll_d", _DEFAULT_START["start_roll_d"])),
        start_pitch_d=float(s.get("pitch_d", _DEFAULT_START["start_pitch_d"])),
        start_yaw_d=float(s.get("yaw_d", _DEFAULT_START["start_yaw_d"])),
    )

    scenes_by_num = {}
    for num, spec in (data.get("scenes") or {}).items():
        moves = (spec or {}).get("moves", []) if isinstance(spec, dict) else (spec or [])
        scenes_by_num[int(num)] = moves
    return start_kwargs, scenes_by_num


# --- Built-in fallback scenes (used only if scenes.yaml can't be read) ---------

_FALLBACK_START = list(_DEFAULT_START.items())

_FALLBACK_REPOSITION = [
    {"type": "right_left", "distance_m": 200.0, "duration_s": TIME_FOR_FIFTY},
    {"type": "forward_backward", "distance_m": 50.0, "duration_s": TIME_FOR_FIFTY},
    {"type": "up_down", "distance_m": 50.0, "duration_s": TIME_FOR_FIFTY},
]

_FALLBACK_SCENES = {
    1: _FALLBACK_REPOSITION + [
        {"type": "forward_backward", "distance_m": -400.0, "duration_s": TIME_FOR_FIFTY * 4},
    ],
    2: _FALLBACK_REPOSITION + [
        {"type": "forward_backward", "distance_m": -150.0, "duration_s": TIME_FOR_FIFTY},
        {"type": "to_point", "lat": 32.20647, "lon": 35.29034, "alt": 530.0,
         "yaw_d": 0.0, "roll_d": 0.0, "pitch_d": 0.0, "duration_s": TIME_FOR_FIFTY * 4},
    ],
}


def _get_definitions():
    """Return ``(start_kwargs, scenes_by_num)``, preferring scenes.yaml."""
    loaded = _load_yaml()
    if loaded is not None:
        return loaded
    return dict(_DEFAULT_START), _FALLBACK_SCENES


def _make_bot(start_kwargs: dict) -> UdpBot:
    """Create + start a target drone bot at the scene spawn pose."""
    bot = UdpBot(udp_port=DRONE_UDP_PORT, send_rate_hz=30.0, **start_kwargs)
    bot.run(blocking=False)
    return bot


def available_scenes():
    """Return the sorted list of defined scene numbers."""
    _, scenes_by_num = _get_definitions()
    return sorted(scenes_by_num)


def run_scene(scene_num: int) -> bool:
    """Play ``scene_num`` (blocking until the trajectory completes).

    Creates and tears down the target bot. Returns True if a scene ran, False if
    the number isn't defined.
    """
    start_kwargs, scenes_by_num = _get_definitions()
    moves = scenes_by_num.get(scene_num)
    if moves is None:
        print(f"[Scenes] scene {scene_num} not found; available: {sorted(scenes_by_num)}")
        return False

    print(f"[Scenes] running scene {scene_num}")
    bot = _make_bot(start_kwargs)
    try:
        for move in moves:
            _apply_move(bot, move)
    finally:
        # Stop and join the background send thread *before* closing the socket.
        # Otherwise the 30 Hz _run_loop keeps calling sendto() on a socket we've
        # already closed -> "[Errno 9] Bad file descriptor".
        bot.stop()
        bot.join(timeout=1.0)
        bot.close()
    print(f"[Scenes] scene {scene_num} finished")
    return True
