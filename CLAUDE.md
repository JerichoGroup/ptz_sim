# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Core 2023 — a PTZ camera simulation framework built on **NVIDIA Isaac Sim 2023.1.1** with **ROS 2 Humble**. Simulates aerial UAV camera views from Cesium 3D Tiles geographic data, driven by GPS/orientation inputs. The `ptz/` module adds a distributed architecture where a Jetson edge device sends PTZ commands to a host PC running Isaac Sim.

## Commands

```bash
# Run simulation locally (requires ISAACSIM_PYTHON on PATH)
ISAACSIM_PYTHON ./simulation/main_sim.py --com-udp --usd-path ./usd/maps/earth/earth.usda

# Convenience wrapper
./run_sim.sh

# Build Docker images
./docker/build_base_image.sh        # warm-start base image with Isaac Sim
./docker/build_simulation_image.sh  # simulation image with extensions

# Launch simulation in Docker (with X11 display)
./run_docker.sh

# Install Python packages
pip install ./simulation/    # installs isaac_core_dev_kit
pip install ./debugger       # installs ros-sender and udp-sender GUI tools

# Debugger GUI tools (after installing debugger package)
ros-sender    # ROS 2-based GUI for sending pose commands
udp-sender    # UDP-based GUI for sending pose commands
```

**Simulation flags** (can be combined):

| Flag | Effect |
|------|--------|
| `--usd-path <file.usda>` | Load specific scene |
| `--com-ros` | Enable ROS 2 comms (mutually exclusive with `--com-udp`) |
| `--com-udp` | Enable UDP comms |
| `--headless` | Run without GUI |
| `--distance-sensor` | Enable laser range sensor |
| `--bbox-publisher` | Publish object bounding boxes |
| `--sat` | Enable screenshot/frame capture |
| `--image-rtp` | Serve camera images over RTSP (H.264, port 8554) |

There are no automated tests or linting setups in this project.

## Architecture

```
simulation/main_sim.py          ← entry point (parses flags, launches Simulation)
simulation/sim_app.py           ← Simulation class (orchestrates everything)
simulation/consts.py            ← all configuration constants (camera, ROS topics, ports)
simulation/sim_utils.py         ← argparse and misc utilities
```

**OmniGraph Extensions** (loaded by Isaac Sim at startup):

| Extension | Role |
|-----------|------|
| `extensions/omni.sim.math/` | Coordinate transforms: WGS84 → ENU local frame |
| `extensions/omni.sim.position/` | Position input nodes (ROS 2 subscriber, UDP receiver) |
| `extensions/omni.sim.sensors/` | Sensor output nodes (ROS 2 publishers, gimbal control) |

**`simulation/isaac_core_dev_kit/`** — installable Python package (`pip install ./simulation/`) providing:
- `isaac_manager/` — `HostIsaacManager` and `DockerIsaacManager` context managers that start/stop Isaac Sim subprocesses and expose a connected `SimApp`
- `core_capture/` — `VideoCapture`, `PoseCapture`, `BboxCapture`, `DistanceCapture` for extracting data from a running sim
- `udp/` — `OnePointSender`, `OrbitSender`, `PathSender` for publishing pose commands; `LLAPoint` dataclass
- `dev_utils/` — helpers: `set_gimbal_angle()`, `save_current_frame_to()`, `delete_cesium_cache()`, `safe_rclpy_init/shutdown()`

**`simulation/script_nodes/`** — Python OmniGraph nodes executed inside Isaac Sim (bbox_node, sat_node, sensor_node).

**`usd/`** — Universal Scene Description assets: maps (Cesium Earth tilesets), cameras (ROS2/UDP variants), sensors, drone assets.

**`ptz/`** — Distributed PTZ simulation (Jetson edge ↔ Isaac Sim host). Everything under `ptz/from_real/` is the real Jetson tracking repo (`dronetracker`) and is **read-only reference — do not modify it**.
- `ptz_sim.py` — host-side `PTZSim` and the main host entry point (`python3 ./ptz/ptz_sim.py`; wraps Isaac via `HostIsaacManager(image_rtp=True)`). Receives UDP commands and drives the gimbal via a **persistent ROS2 publisher** on `/isaac_core/gimbal` (+ `/isaac_core/zoom`). Background threads: `_recv_loop` (parse commands), `_pose_loop` (20 Hz pose back), `_mover_loop` (25 Hz TTL-coalesced ContinuousMove integrator), `_zoom_loop` (20 Hz zoom slew, `netz250.ZOOM_FULL_TRAVEL_S` = 4 s full travel), `_ros_spin_loop` (bbox telemetry). The target drone runs its own threads inside `drone.py` (`SimDrone`), started on demand. `relative_move` uses ONVIF **TranslationSpaceFov** units: `dp`/`dt` are fractions of the *current* (zoom-dependent) HFoV, matching `controller.py` — not a constant angle. `stop_all` halts motion but leaves the sim running; home is auto-saved at the startup pose. Camera look-point set in `_init_camera_position` (lat 32.20647, lon 35.29034, alt 540, yaw 90). Env vars: `PTZ_JETSON_IP`, `PTZ_JETSON_PORT` (default 5006), `PTZ_DRONE_SPEED_MS` (default 10).
- `ptz_sim_controller.py` — Jetson-side `PTZSimController`: drop-in replacement for `PTZController` (same public API — `move`/`relative_move`/`zoom_*`/`center_and_zoom`/`vel_scale`/`save_home`/`go_home`/`trigger_autofocus`). Serialises commands to JSON over UDP; runs a 1.5 s `ping` heartbeat; syncs sim zoom truth back into `zoom_pos` from received pose. **Note:** this is a standalone copy used only by this repo's own manual test tools (`ptz_tui.py`, `jetson_test_ptz_sim.py`) — the real Jetson pipeline uses `dronetracker/ptz/sim_controller.py` from the `DroneTracker` repo itself (via `sim_live_ptz.py`), which is the actively-maintained one; that copy sends a `session` token on the heartbeat (no `scene` field — scenes are keyboard-driven now) and carries the `scene`/`drone_vel`/`drone_stop` messages. Keep `vel_scale()` in sync between the two (fixed 2026-07-28: both now use the real controller's linear formula, not the older FOV-ratio one).
- `scenes.py` — scene library. A *scene* flies a simulated target drone (`UdpBot`, sent to Isaac on **UDP 33335**) through a canned trajectory so the Jetson tracker has something to lock onto. Scenes are **declarative** — a list of `(move, metres)` steps in the `SCENES` dict, editable without Python knowledge; `validate()` reports mistakes. Directions are camera-relative. The target does not exist until a scene or manual flight asks for one, then spawns at `SPAWN_OFFSET` (200 m forward, 40 m up) from the camera — spawning it *at* the camera used to black out the video.
- `from_real/dronetracker/ptz/controller.py` — real ONVIF `PTZController`; fidelity reference for the two files above.
- `jetson_test_ptz_sim.py` — standalone Jetson test driver (cycles through commands). `ptz_tui.py` — curses keyboard controller (`--host <HOST_IP>`, `m` toggles remote/local target). `rtsp_test.py` — RTSP viewer (`--backend gst` = low-latency, `--backend ffmpeg`). `host_udp_listener.py` / `nirchuk.py` — debug scratch scripts (raw UDP dump; bot-flying prototype behind `scenes.py`).

**`from_real/dronetracker` refresh (2026-07-28):** this reference copy was stale by a wide margin — well beyond the `feature/ptz_sim` branch's own 23-commit rebase gap onto `DroneTracker` `main`. It predated the entire auto-hunt/lock-on/follow/appearance/distance/filtering/template-track feature set, the `pipeline/detect/`+`pipeline/display/` split, the segmented storage system, `go_home` generation-cancellation (`_go_home_gen`/`_homing`), the `autofocus_once()` rename, and the `vel_scale()` linear-formula change. Re-synced wholesale from the current, rebased `DroneTracker` `feature/ptz_sim` (which itself now sits on current `main`) — a plain directory replace (not a submodule), excluding `dronetracker/ptz/sim_controller.py` and `dronetracker/apps/sim_live_ptz.py` (those are DroneTracker's own **sim-side** additions, not real-camera code, so they don't belong in a "real interface" reference).

**Interface audit findings from the refresh** — confirmed the wire protocol below is **unchanged and still correct** against the refreshed reference; no `ptz_sim.py`/`scenes.py` logic needed updating:
- `PTZController.relative_move()` **no longer exists as a standalone method** in current `main` — it was folded into `center_and_zoom()`'s internal RelativeMove+zoom combo. This doesn't affect the sim: the live pipeline (`dronetracker/pipeline/detect/idle.py`) only ever calls `center_and_zoom()`, never a standalone `relative_move`, so `ptz_sim.py`'s `relative_move` UDP handler (`_apply_relative`) is unreachable dead code from the real pipeline's perspective — harmless to leave as-is.
- `center_and_zoom`'s `delta_pan`/`delta_tilt`/`zoom_target`/`space`/`parallel` signature and its `TranslationSpaceFov` semantics are unchanged; `dronetracker/ptz/lockon.py`'s `compute_lockon_command()` still emits `dx`/`dy` as `±1.0`-clipped fractions of the current FOV (`frozen.fov_gain * 2.0 * ex`), matching `ptz_sim.py`'s `_apply_relative()` HFoV-fraction assumption exactly.
- `autofocus_once()` (public rename of the old private `_autofocus()`) is called **only internally** by the real controller (from its own zoom-burst/go-home helpers), never from the pipeline layer — so it has no host-side (`ptz_sim.py`) equivalent to add.
- **Net result: no functional changes were needed in `ptz_sim.py`, `scenes.py`, or the UDP wire protocol.** This was a documentation/reference-fidelity refresh only.


**Communication topology (NAT/VPN-aware, reply-to-sender):** the Jetson sits behind a NAT/VPN bridge, so the host can never *initiate* to it. Both ends bind ONE UDP socket; the host replies pose to the **source address** of received packets (`jetson_ip=None` default — set env `PTZ_JETSON_IP` only for a directly-routable setup). A 1.5 s `ping` heartbeat from the Jetson keeps the NAT mapping open and the reply address fresh.
```
Jetson (PTZSimController, binds :5006) ──cmd UDP──▶ Host (PTZSim, binds :5005)
                                       ◀─pose UDP── (reply-to-sender, 20 Hz, same socket)
Host (UdpBot target) ──────UDP :33335──▶ Isaac Sim (drone prim)   [scene playback]
Host RTSP H.264 (:8554) ───────────────video──────▶ Jetson RtspGrabber
```

**Command protocol (JSON over UDP):** every packet is `{"type": <str>, "time": <float>, ...}`.
- Jetson→host: `move {vx,vy}`, `relative_move {dp,dt,space}`, `zoom_to {target}`, `center_and_zoom {dp,dt,zoom_target,...}`, `save_home`, `go_home`, `stop_all`, and `ping` (heartbeat).
- host→Jetson: `pose {pan,tilt,zoom,timestamp}` at 20 Hz.
- **Simulator-only fields** on the pose packet (read off *any* packet): `drone` `{fwd,right,up,range}` in camera-relative metres, `scene` (which scene is playing, or null), `joystick` (manual flight active). The Jetson's `SimControls` consumes these via `PTZSimController.extra_msg_handler`, which is the controller's ONLY extra public attribute. Commands in: `scene{n}` (n=0 ⇒ manual flight), `drone_vel{fwd,right,up}`, `drone_stop`. `session` re-arms nothing now — scenes are triggered by keypress, not by session token.

**Video (RTSP) + realism effects:** `simulation/libraries/ros_image_to_rtp_lib.py` serves H.264 over RTSP via `GstRtspServer` (port `RTSP_PORT=8554`; paths `/unicast/c1/s0/live` & `/cam/realmonitor`), launched as a subprocess by `lib_manager.py` when `--image-rtp`. The `format=I420` cap is load-bearing (RGB→4:4:4 breaks baseline H.264 → corrupt output). The server is low-latency (verified via a `gst-launch` client); OpenCV+FFMPEG clients add ~2 s of their own RTSP buffering, so use a GStreamer client (`appsink sync=false drop=true max-buffers=1`) for low latency. Before encoding, a configurable `FrameEffects` layer (the `EFFECTS` / `SpriteKind` dataclasses at the top of the file) adds realism: a **zoom-refocus blur** (Gaussian, keyed to `/isaac_core/zoom` changes, peaks ~0.8 s and clears by `blur_time`=2 s) and **drifting sprites** — small fast flies (static PNGs from `pics/`, 0–2 on screen) and bigger slower **animated** birds (GIFs from `gifs/`, frames played back as wing-flaps, 0–1 on screen at ~half the fly rate).

**Coordinate systems used throughout:**
- WGS84 (lat/lon/alt) — external input
- ENU (East-North-Up) — Isaac Sim local frame
- NED (North-East-Down) — MAVRos/aviation convention, converted at boundary

## Key Configuration (`simulation/consts.py`)

```python
RESOLUTION_WIDTH, RESOLUTION_HEIGHT = 1920, 1080
CAMERA_FOV = 78.1           # degrees
TILESETS_HTTP_SERVER_URL = "http://10.20.15.122:8088"   # Cesium tile server
MAX_OUTPUTS_ROS_HRZ = 30    # Hz
# Key ROS topics
GLOBAL_POSE_TOPIC = "/isaac_core/global_pose"
IMAGE_TOPIC       = "/isaac_core/image_rgb"
GIMBAL_TOPIC      = "/isaac_core/gimbal"
ZOOM_TOPIC        = "/isaac_core/zoom"      # Float32, drives zoom_node FOV
LASER_TOPIC       = "/isaac_core/distance_sensor"
# RTSP video (ptz/ pipeline)
RTSP_PORT  = 8554
RTSP_PATHS = ["/unicast/c1/s0/live", "/cam/realmonitor"]
RTP_VIDEO_PORT = 5004   # legacy, unused
RTP_META_PORT  = 5005   # legacy, unused
```

PTZ UDP ports: host `PTZSim` binds **5005** (commands in); Jetson `PTZSimController` binds **5006** (pose in). Pose is returned reply-to-sender on the host's command socket (NAT/VPN-aware). Scene playback sends the target drone to Isaac on **33335** (`scenes.DRONE_UDP_PORT`).
