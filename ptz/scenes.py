"""Scene library for the PTZ simulator.

A *scene* flies a simulated target (a ``UdpBot`` drone) through a trajectory so
the Jetson's tracker has something to detect and lock onto. The Jetson selects a
scene by number via its config (``sim.scene``); that number is carried over the
existing PTZ command channel, and ``ptz_sim`` plays the scene a configurable
delay after Isaac Sim finishes loading.

Adding a scene:
  1. Write a ``def scene_N(bot): ...`` that drives ``bot`` (a fresh ``UdpBot``).
  2. Register it in ``SCENES`` below.
The bot is created, started, and closed for you by ``run_scene`` — a scene body
just issues movement commands (which block for their ``duration_s``).
"""

import time  # noqa: F401  (handy for future scenes that want explicit pauses)

from isaac_core_dev_kit.udp.udp_bot import UdpBot


# UDP port Isaac Sim listens on for the target/drone prim (matches nirchuk.py).
DRONE_UDP_PORT = 33335

# Target spawn pose (lat, lon, alt + orientation in degrees).
_START = dict(
    start_lat=32.20647, start_lon=35.29034, start_alt=540.0,
    start_roll_d=0.0, start_pitch_d=0.0, start_yaw_d=0.0,
)

# Base time unit (seconds) used to pace the canned moves.
TIME_FOR_FIFTY = 2.0


def _make_bot() -> UdpBot:
    """Create + start a target drone bot at the scene spawn pose."""
    bot = UdpBot(udp_port=DRONE_UDP_PORT, send_rate_hz=30.0, **_START)
    bot.run(blocking=False)
    return bot


def _drone_to_start(bot: UdpBot) -> None:
    """Common opening move shared by the scenes (positions the target)."""
    bot.move_right_left(distance_m=200.0, duration_s=TIME_FOR_FIFTY)
    bot.move_forward_backward(distance_m=50.0, duration_s=TIME_FOR_FIFTY)
    bot.move_up_down(distance_m=50.0, duration_s=TIME_FOR_FIFTY)


def scene_1(bot: UdpBot) -> None:
    """Target repositions, then makes a long straight run away from the camera."""
    _drone_to_start(bot)
    bot.move_forward_backward(distance_m=-400.0, duration_s=TIME_FOR_FIFTY * 4)


def scene_2(bot: UdpBot) -> None:
    """Target repositions, advances, then descends toward a fixed point."""
    _drone_to_start(bot)
    bot.move_forward_backward(distance_m=-150.0, duration_s=TIME_FOR_FIFTY)
    bot.move_to_point(
        target_lat=32.20647, target_lon=35.29034, target_alt=530.0,
        target_yaw_d=0.0, target_roll_d=0.0, target_pitch_d=0.0,
        duration_s=TIME_FOR_FIFTY * 4,
    )


# Scene registry — map config `sim.scene` numbers to scene functions.
SCENES = {
    1: scene_1,
    2: scene_2,
}


def available_scenes():
    """Return the sorted list of registered scene numbers."""
    return sorted(SCENES)


def run_scene(scene_num: int) -> bool:
    """Play ``scene_num`` (blocking until the trajectory completes).

    Creates and tears down the target bot. Returns True if a scene ran, False if
    the number isn't registered.
    """
    fn = SCENES.get(scene_num)
    if fn is None:
        print(f"[Scenes] scene {scene_num} not found; available: {available_scenes()}")
        return False

    print(f"[Scenes] running scene {scene_num}")
    bot = _make_bot()
    try:
        fn(bot)
    finally:
        bot.close()
    print(f"[Scenes] scene {scene_num} finished")
    return True
