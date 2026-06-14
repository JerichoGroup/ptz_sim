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
| `--image-rtp` | Stream images over RTP |

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

**`ptz/`** — Distributed PTZ simulation (newest module):
- `ptz_sim.py` — host-side: listens for UDP commands from Jetson, controls gimbal via `set_gimbal_angle()`, sends pose back at 20 Hz
- `ptz_sim_controller.py` — Jetson-side: drop-in replacement for real PTZ controller, sends commands over UDP
- `controller.py` — real ONVIF PTZ controller (Jetson only, has dronetracker imports)
- `jetson_test_ptz_sim.py` — Jetson-side test entry point

**Communication topology:**
```
Jetson (PTZSimController) ──UDP:5005──▶ Host (PTZSim / Isaac Sim)
                          ◀──UDP:5006── (pose feedback at 20 Hz)
```

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
LASER_TOPIC       = "/isaac_core/distance_sensor"
RTP_VIDEO_PORT = 5004
RTP_META_PORT  = 5005
```
