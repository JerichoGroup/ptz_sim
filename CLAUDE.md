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

**`ptz/`** — Distributed PTZ simulation (Jetson edge ↔ Isaac Sim host). Everything under `ptz/from_real/` is the real Jetson tracking repo (`dronetracker`) and is treated as **read-only reference — do not modify it**.
- `ptz_sim.py` — host-side `PTZSim`: receives UDP commands and drives the gimbal via a **persistent ROS2 publisher** on `/isaac_core/gimbal` (+ `/isaac_core/zoom`). Runs a 25 Hz ContinuousMove integrator (TTL-coalesced), a 20 Hz zoom slew (7 s full travel), and a cosine `go_home` slew. `relative_move` uses ONVIF **TranslationSpaceFov** units: `dp`/`dt` are fractions of the *current* (zoom-dependent) HFoV, matching `controller.py` — not a constant angle. `stop_all` halts motion but leaves the sim running; home is auto-saved at the startup pose.
- `ptz_sim_controller.py` — Jetson-side `PTZSimController`: drop-in replacement for `PTZController` (same public API — `move`/`relative_move`/`zoom_*`/`center_and_zoom`/`vel_scale`/`save_home`/`go_home`/`trigger_autofocus`). Serialises commands to JSON over UDP.
- `from_real/dronetracker/ptz/controller.py` — the real ONVIF `PTZController`; fidelity reference for the two files above.
- `jetson_test_ptz_sim.py` — standalone Jetson test driver. `ptz_tui.py` — curses keyboard controller (`--host <HOST_IP>`, `m` toggles remote/local). `rtsp_test.py` — RTSP viewer (`--backend gst` = low-latency, `--backend ffmpeg`).

**Communication topology (NAT/VPN-aware, reply-to-sender):** the Jetson sits behind a NAT/VPN bridge, so the host can never *initiate* to it. Both ends bind ONE UDP socket; the host replies pose to the **source address** of received packets (`jetson_ip=None` default — set env `PTZ_JETSON_IP` only for a directly-routable setup). A 1.5 s `ping` heartbeat from the Jetson keeps the NAT mapping open and the reply address fresh.
```
Jetson (PTZSimController, binds :5006) ──cmd UDP──▶ Host (PTZSim, binds :5005)
                                       ◀─pose UDP── (reply-to-sender, 20 Hz, same socket)
Host RTSP H.264 (:8554) ───────────────video──────▶ Jetson RtspGrabber
```

**Video (RTSP):** `simulation/libraries/ros_image_to_rtp_lib.py` serves H.264 over RTSP via `GstRtspServer` (port `RTSP_PORT=8554`; paths `/unicast/c1/s0/live` & `/cam/realmonitor`), launched as a subprocess by `lib_manager.py` when `--image-rtp`. The `format=I420` cap is load-bearing (RGB→4:4:4 breaks baseline H.264 → corrupt output). The server is low-latency (verified via a `gst-launch` client); OpenCV+FFMPEG clients add ~2 s of their own RTSP buffering, so use a GStreamer client (`appsink sync=false drop=true max-buffers=1`) for low latency.

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

PTZ UDP ports: host `PTZSim` binds **5005** (commands in); Jetson `PTZSimController` binds **5006** (pose in). Pose is returned reply-to-sender on the host's command socket (NAT/VPN-aware).
