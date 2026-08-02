"""Scene library — the canned flight paths the simulated target drone can fly.

═══════════════════════════════════════════════════════════════════════════════
 HOW TO ADD OR CHANGE A SCENE  (no programming needed)
═══════════════════════════════════════════════════════════════════════════════

A scene is just a numbered list of moves.  Copy an existing block, change the
number, the name and the moves.  Press the matching number key (1-9) in the
DroneTracker window to fly it.

Every move is one line:

    {"move": "forward", "meters": 100, "seconds": 4},

  "move"    – which direction (see the table below)
  "meters"  – how far to travel
  "seconds" – how long the move should take (so 100 m in 4 s = 25 m/s)

Directions are **relative to the camera**, exactly like the manual flight keys:

    forward    away from the camera        backward   toward the camera
    right      to the camera's right       left       to the camera's left
    up         gain altitude               down       lose altitude

Two extra moves are available:

    {"move": "wait", "seconds": 3}                       – hover in place
    {"move": "goto", "lat": 32.2065, "lon": 35.2903,
     "alt": 540, "seconds": 5}                           – fly to a fixed point

The drone always starts a scene at the same place — a fixed distance in front of
the camera (see SPAWN_OFFSET below) — so scenes are repeatable.

Tips
  • Small "seconds" for a fast pass, large for a slow crawl.
  • A drone flying "forward" gets smaller/harder to see — good for range tests.
  • Keep total time sensible; the scene ends when the last move finishes and the
    drone then just hovers where it stopped.
═══════════════════════════════════════════════════════════════════════════════
"""

# UDP port Isaac Sim listens on for the target drone prim (/bboxes/full_drone).
DRONE_UDP_PORT = 33335

# Where the camera itself sits and which way it faces.  ptz_sim uses this to
# place the camera, so it is the single source of truth for the scene geography.
CAMERA = {
    "lat": 32.20647,
    "lon": 35.29034,
    "alt": 540.0,
    "yaw_deg": 90.0,
}

# Where the drone appears, measured FROM THE CAMERA and relative to where it is
# looking.  It must not be (0, 0, 0): the drone would spawn inside the lens and
# the video would go black.
#
#   forward – metres away from the camera        right – metres to its right
#   up      – metres above it
SPAWN_OFFSET = {
    "forward": 200.0,
    "right": 0.0,
    "up": 40.0,
}


# ═══════════════════════════════════════════════════════════════════════════
#  THE SCENES — edit freely
# ═══════════════════════════════════════════════════════════════════════════

SCENES = {
    1: {
        "name": "reposition then long run away",
        "steps": [
            {"move": "right",    "meters": 200, "seconds": 2},
            {"move": "forward",  "meters": 50,  "seconds": 2},
            {"move": "up",       "meters": 50,  "seconds": 2},
            {"move": "backward", "meters": 400, "seconds": 8},
        ],
    },

    2: {
        "name": "reposition, advance, then descend to a fixed point",
        "steps": [
            {"move": "right",    "meters": 200, "seconds": 2},
            {"move": "forward",  "meters": 50,  "seconds": 2},
            {"move": "up",       "meters": 50,  "seconds": 2},
            {"move": "backward", "meters": 150, "seconds": 2},
            {"move": "goto",     "lat": 32.20647, "lon": 35.29034,
             "alt": 530.0, "seconds": 8},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  Below here is machinery — you do not need to touch it to add a scene.
# ═══════════════════════════════════════════════════════════════════════════

# Camera-relative unit vectors: (forward, right, up) multipliers per move name.
MOVE_AXES = {
    "forward":  (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "right":    (0.0, 1.0, 0.0),
    "left":     (0.0, -1.0, 0.0),
    "up":       (0.0, 0.0, 1.0),
    "down":     (0.0, 0.0, -1.0),
}


def available_scenes():
    """Sorted list of scene numbers that are defined."""
    return sorted(SCENES)


def validate():
    """Check every scene for obvious mistakes; returns a list of problems.

    This is the safety net for editing SCENES by hand, so it must never raise —
    a malformed entry has to come back as a readable problem, not a traceback.
    Called at host startup (``ptz/ptz_sim.py``), which prints whatever it returns.
    """
    problems = []
    for num, scene in SCENES.items():
        if not isinstance(num, int) or not 1 <= num <= 9:
            problems.append(f"scene key {num!r} must be a whole number 1-9")
        if not isinstance(scene, dict):
            problems.append(f"scene {num} must be a block with 'name' and 'steps'")
            continue
        if "steps" not in scene or not scene["steps"]:
            problems.append(f"scene {num} has no steps")
            continue
        for i, step in enumerate(scene["steps"], 1):
            if not isinstance(step, dict):
                problems.append(
                    f"scene {num} step {i} is {type(step).__name__}, not a block — "
                    f"each step looks like {{'move': 'forward', 'meters': 100, "
                    f"'seconds': 10}}")
                continue
            move = step.get("move")
            if move in MOVE_AXES:
                for k in ("meters", "seconds"):
                    if k not in step:
                        problems.append(
                            f"scene {num} step {i} ({move}) needs '{k}'")
                    elif not isinstance(step[k], (int, float)):
                        problems.append(
                            f"scene {num} step {i} ({move}): '{k}' must be a "
                            f"number, got {step[k]!r}")
            elif move == "wait":
                if "seconds" not in step:
                    problems.append(f"scene {num} step {i} (wait) needs 'seconds'")
            elif move == "goto":
                for k in ("lat", "lon", "alt", "seconds"):
                    if k not in step:
                        problems.append(f"scene {num} step {i} (goto) needs '{k}'")
            else:
                problems.append(
                    f"scene {num} step {i}: unknown move {move!r} — "
                    f"use one of {sorted(MOVE_AXES)} or 'wait'/'goto'")
    return problems
