# PTZ Camera Simulation — Isaac Sim ↔ Jetson

This repo simulates a **real PTZ security camera** so the drone-tracking algorithm can be
developed and tested without the physical camera.

From the algorithm's point of view nothing changes: it still receives an **RTSP H.264 video
stream** and still sends the **same pan / tilt / zoom commands** it would send to the real
camera. The difference is that behind the scenes those commands move a virtual camera inside
an **NVIDIA Isaac Sim** world, and the video is rendered rather than captured.

Two machines are involved:

| Machine | Nickname | Runs | Repo |
|---|---|---|---|
| Workstation with an NVIDIA GPU | **the sim stand** (host) | Isaac Sim + the PTZ command server + the RTSP video server | `ptz_sim` (this repo) |
| NVIDIA Jetson | **the Jetson** | The real tracking pipeline (VMD + YOLO + PTZ control) | `DroneTracker`, branch `feature/ptz_sim` |

---

## 1. How the communication works

There are **four** independent flows. Three of them cross the machine boundary, and one is
internal to the sim stand.

```
                    SIM STAND (host)                                    JETSON
        ┌──────────────────────────────────────────┐          ┌──────────────────────────┐
        │                                          │          │                          │
        │   ptz/ptz_sim.py   (PTZSim)              │          │  sim_motion_ptz_pipeline │
        │   binds UDP :5005  ◄─────────────────────┼──① cmds──┤  PTZSimController        │
        │                    ──────────────────────┼──② pose──►  binds UDP :5006         │
        │          │                               │          │            │             │
        │          │ ROS 2 topics                   │          │            │ frames      │
        │          ▼                               │          │            ▼             │
        │   ┌──────────────────┐                   │          │      RtspGrabber         │
        │   │    Isaac Sim     │  /isaac_core/     │          │            ▲             │
        │   │  (virtual camera)│   gimbal, zoom    │          │            │             │
        │   └──────────────────┘                   │          │            │             │
        │          │ /isaac_core/image_rgb         │          │            │             │
        │          ▼                               │          │            │             │
        │   ros_image_to_rtp_lib (GStreamer)       │          │            │             │
        │   RTSP server :8554  ────────────────────┼──③ video─┼────────────┘             │
        │          ▲                               │          │                          │
        │          │ ④ target drone UDP :33335     │          │                          │
        │   ptz/scenes.py (UdpBot)                 │          │                          │
        └──────────────────────────────────────────┘          └──────────────────────────┘
```

### ① Commands — Jetson → sim stand (UDP, JSON)

The Jetson sends one small JSON packet per command to **`sim.host_ip:5005`**. Every packet
looks like `{"type": "<name>", "time": <unix-seconds>, ...}`:

| Command | Payload | Meaning |
|---|---|---|
| `move` | `vx`, `vy` | Continuous pan/tilt velocity (like ONVIF *ContinuousMove*). Re-sent while a key is held; expires after `cmd_ttl` (0.15 s) so the camera stops if packets stop. |
| `relative_move` | `dp`, `dt`, `space` | One-shot nudge, in **fractions of the current field of view** (±1.0 = half the current HFoV). |
| `center_and_zoom` | `dp`, `dt`, `zoom_target`, `space`, `parallel` | The lock-on move: centre the target **and** zoom in, together. This is the one the tracker actually uses. |
| `zoom_to` | `target` | Absolute zoom, `0.0` = widest … `1.0` = full tele. |
| `save_home` | – | Remember the current pose as "home". |
| `go_home` | – | Slew back to the saved home pose and zoom. |
| `stop_all` | – | Stop all motion. **Does not shut the simulator down** — the Jetson sends this on every quit, and Isaac Sim takes too long to start to throw it away. |
| `ping` | `scene`, `session` | Heartbeat, every 1.5 s. See below. |

Because `dp`/`dt` are FoV fractions rather than fixed angles, a nudge automatically becomes
finer as you zoom in — exactly how the real camera's `TranslationSpaceFov` behaves.

### ② Pose — sim stand → Jetson (UDP, JSON, 20 Hz)

The sim replies `{"type":"pose","pan":…,"tilt":…,"zoom":…,"timestamp":…}` twenty times a
second, so the Jetson always knows where the camera actually is (it drives the HUD and keeps
the zoom value honest).

The important design point: **the sim never initiates traffic to the Jetson.** It replies to
the *source address of whatever packet it last received* ("reply-to-sender"), sent from the
very same socket the commands arrived on. That's why:

* there is **no Jetson IP to configure anywhere** — it works through NAT, a VPN, or the
  plain USB link without changes;
* the 1.5 s `ping` heartbeat matters even when nobody is touching the controls — it keeps
  the return path fresh so pose keeps flowing while idle.

If you ever have a directly routable setup and want to hard-code the target instead, set the
`PTZ_JETSON_IP` env var on the sim stand.

### ③ Video — sim stand → Jetson (RTSP / H.264 over TCP)

Isaac Sim publishes rendered frames on the ROS 2 topic `/isaac_core/image_rgb`
(1920×1080). `simulation/libraries/ros_image_to_rtp_lib.py` encodes them to H.264 with
GStreamer and serves them over RTSP on **port 8554**, at **both** paths the real camera
uses:

```
/unicast/c1/s0/live      ← the primary URL
/cam/realmonitor         ← the fallback URL
```

**The Jetson opens this connection**, so the return path is never a problem.

A real camera serves RTSP on the privileged port **554**, and Linux won't let a normal user
bind anything below 1024. So the server binds 8554 and we redirect 554 → 8554 with one
`iptables` rule (see [§2](#2-one-time-setup-on-the-sim-stand)). The payoff: the Jetson's
config needs **no simulator-specific URL at all** — sim and real camera differ only by
`camera.ip`.

Before encoding, a small **realism layer** is applied: a Gaussian *refocus blur* that ramps
up and clears after each zoom change (mimicking the real lens hunting focus), plus drifting
sprites — small fast "flies" and larger animated "birds" — so the detector sees the kind of
clutter it meets in the field. Everything is tunable via the `EFFECTS` dataclass at the top
of that file.

### ④ Target drone — internal to the sim stand (UDP :33335)

A *scene* flies a simulated target drone through a scripted trajectory so the tracker has
something to lock onto. `ptz/scenes.py` sends the target's pose to Isaac Sim on UDP 33335.

You choose the scene **from the Jetson**, via `sim.scene` in its `config.yaml`. The number
rides along on the heartbeat, together with a `session` token that is regenerated every
time the Jetson pipeline starts. The sim plays the requested scene **once per new session**,
`PTZ_SCENE_DELAY_S` (default 20 s) after Isaac has loaded. Practical consequence:

> Quit the Jetson pipeline → change `sim.scene` → run it again, and the scene replays.
> **You never have to restart Isaac Sim to change or repeat a scene.**

Registered scenes:

| `sim.scene` | What the target does |
|---|---|
| `0` | Nothing — no target is spawned. |
| `1` | Repositions, then makes a long straight run away from the camera. |
| `2` | Repositions, advances, then descends toward a fixed point. |

Add your own by writing a `scene_N(bot)` function and registering it in the `SCENES` dict in
`ptz/scenes.py`.

### Port summary

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 5005 | UDP | Jetson → sim | PTZ commands |
| 5006 | UDP | sim → Jetson | Pose replies (reply-to-sender) |
| 554 → 8554 | TCP | Jetson → sim | RTSP video (554 redirected to 8554) |
| 33335 | UDP | internal to sim | Target drone pose → Isaac Sim |

---

## 2. One-time setup on the sim stand

**Redirect the RTSP port** so the Jetson can use the standard camera URL. Run this once per
boot (it does not survive a reboot):

```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 554 -j REDIRECT --to-port 8554
```

Check it is there:

```bash
sudo iptables -t nat -L PREROUTING -n --line-numbers | grep 8554
```

<details>
<summary>Prefer not to touch iptables?</summary>

You can instead point the Jetson straight at port 8554 by setting these in the Jetson's
`config.yaml`, but then the Jetson config is no longer identical to the real-camera one:

```yaml
camera:
  rtsp_main_override: "rtsp://<SIM_IP>:8554/unicast/c1/s0/live"
  rtsp_alt_override:  "rtsp://<SIM_IP>:8554/cam/realmonitor"
```
</details>

**Install the Python packages** used by the dev-kit and the debug tools (once):

```bash
pip install ./simulation      # installs isaac_core_dev_kit
pip install ./debugger        # installs the ros-sender / udp-sender GUI tools
```

## 3. One-time setup on the Jetson

The `DroneTracker` repo has a helper that installs the correct **JetPack-matched** PyTorch
plus the app dependencies. This matters: a plain `pip install torch` pulls an x86 wheel that
reports a version but cannot see the GPU.

```bash
cd ~/clones/DroneTracker
git checkout feature/ptz_sim
./setup_jetson_env.sh          # add --yes to skip the prompts
```

Then check `config.yaml` on the Jetson:

```yaml
camera:
  ip: "<SIM_STAND_IP>"         # the sim stand — this is the only sim/real difference
sim:
  host_ip: "<SIM_STAND_IP>"
  host_port:   5005
  listen_port: 5006
  scene: 0                     # 0 = no target, 1 or 2 = play that scene
yolo:
  model_path: "models/yolo26n.engine"   # must exist on the Jetson
```

> A TensorRT `.engine` only loads on the exact TensorRT version that built it, so build
> engines **on the Jetson** (`python scripts/export_engine.py`).

---

## 4. Running the system

Two terminals, two machines. **Start the sim stand first** — Isaac Sim takes ~15 s to load.

### Step 1 — on the sim stand

```bash
cd ~/clones/ptz_sim
ISAACSIM_PYTHON ./ptz/ptz_sim.py
```

You should see:

```
[Host] PTZSim listening on 0.0.0.0:5005
[Host] Jetson pose target: reply-to-sender (source of received commands)
IsaacSim loaded in 14.02 seconds!
[Host] Isaac Sim started.
[Host] Home saved: pan=0.00, tilt=0.00, zoom=0.00
```

The RTSP server's own log lives in a file, because it runs in a subprocess whose output is
hidden by default. Watch it in a second terminal — this is the single most useful thing to
look at when video misbehaves:

```bash
tail -f /tmp/ptz_sim_rtsp.log
```

Healthy output looks like this (`image_topic` = frames arriving from Isaac, `pushed` =
frames going to a connected client):

```
[RTSP] Mounted at /unicast/c1/s0/live
[RTSP] Mounted at /cam/realmonitor
[RTSP] Server listening on :8554
[RTSP] no client: image_topic=25.0 fps  pushed=0.0 fps
[RTSP] Client connected — pipeline started
[RTSP] client attached: image_topic=25.0 fps  pushed=25.0 fps
```

### Step 2 — on the Jetson

```bash
cd ~/clones/DroneTracker
python3 ./sim_motion_ptz_pipeline.py
```

A window opens with the simulated camera view, and the tracker runs against it exactly as it
would against the real camera.

> Needs a real display. Over SSH, prefix with `DISPLAY=:0` to put the window on the Jetson's
> attached monitor.

### Other entry points

| Where | Command | What it does |
|---|---|---|
| Jetson | `python3 ./run_live_ptz.py` | The **real camera** pipeline (unchanged by any of this). |
| Sim stand | `python3 ptz/rtsp_test.py` | Just view the stream — handy for isolating video from the tracker. Add `--backend ffmpeg` to use the same decoder the Jetson uses. |
| Sim stand | `python3 ptz/ptz_tui.py --host <SIM_IP>` | Drive the camera by keyboard without running the tracker. |
| Sim stand | `python3 ptz/jetson_test_ptz_sim.py` | Fire a scripted sequence of PTZ commands. |

### Keyboard controls (in the pipeline window)

| Key | Action |
|---|---|
| `T` / `Enter` | Lock on to the selected target (IDLE → TRACK) |
| `R` | Return to home pose |
| `H` | Save the current pose as home |
| Arrow keys | Pan / tilt (IDLE only) |
| `Z` / `X` | Zoom in / out |
| `V` | Start / stop recording |
| `S` | Save a screenshot |
| `Q` | Quit |

### Shutting down

Quit the Jetson pipeline with `Q` first, then stop the sim stand with `Ctrl-C`. The Jetson's
`stop_all` deliberately does **not** stop the simulator, so you can restart the Jetson side
as often as you like while Isaac Sim stays up.

---

## 5. Troubleshooting

### The Jetson prints `DESCRIBE failed: 503 Service Unavailable`

The RTSP server is reachable but cannot serve video. Check `/tmp/ptz_sim_rtsp.log`:

* **`FATAL: could not bind RTSP port 8554`** — a **stale `lib_manager` from a previous run is
  still holding the port**. This is the classic one: `lib_manager` is a subprocess of Isaac
  Sim, and if Isaac is killed rather than exited cleanly it can be left behind, keeping 8554
  bound. Your new server then serves nobody while the dead one answers the Jetson with 503.

  ```bash
  ss -ltnp | grep 8554                              # see who holds it
  pkill -9 -f 'ptz_sim/simulation/lib_manager'      # clear it
  ```

  `main_sim.py` now asks the kernel to kill `lib_manager` when Isaac Sim dies, so this should
  not recur — but it is still the first thing to check.

* **`WARNING: no frames from the image topic`** — the server is up but Isaac isn't delivering
  frames, so there is nothing to encode. Confirm the camera is publishing:

  ```bash
  ros2 topic hz /isaac_core/image_rgb
  ```

* **`image_topic=25 fps` but `no client` forever** — frames are fine and the Jetson never
  actually reached *this* server. Almost always the stale-port case above.

### The Jetson prints `404 Not Found`

The path is wrong. Only `/unicast/c1/s0/live` and `/cam/realmonitor` are mounted. Clear any
`rtsp_main_override` / `rtsp_alt_override` in the Jetson's `config.yaml`.

### The Jetson prints `Connection refused` on port 554

The `iptables` redirect isn't in place — see [§2](#2-one-time-setup-on-the-sim-stand). It is
lost on reboot.

### Commands work but there's no video (or vice-versa)

They are completely independent flows: commands are UDP 5005, video is TCP 554/8554. A
working scene playback proves only that the **command** path is healthy.

### `CUDA initialization failure` / `no CUDA-capable device` on the Jetson

A non-Jetson PyTorch wheel got installed. Re-run `./setup_jetson_env.sh`; a correct install
reports a `+cu126` version and passes the script's final `torch.cuda` check.

### `[Storage] ERROR: could not create chunk dir: Permission denied`

Recording only — the tracker still runs. Give the Jetson user ownership of the storage path
(`storage.base_dir` in `config.yaml`), e.g. `sudo chown -R $USER /mnt/ssd/dronetracker`.

### Cesium tile errors on the sim stand

Lines like `An unexpected error occurred when loading tile: curl: Couldn't connect to server`
refer to the terrain tile server and are unrelated to PTZ or video. Safe to ignore.

---

## 6. Where things live

```
ptz/
  ptz_sim.py               Host entry point. UDP command server + gimbal/zoom
                           publishing + pose replies + scene playback.
  scenes.py                Target-drone trajectories (the SCENES registry).
  ptz_sim_controller.py    Stand-alone copy of the Jetson-side controller, used only
                           by the local debug tools below. The pipeline itself uses
                           DroneTracker's own dronetracker/ptz/sim_controller.py.
  ptz_tui.py               Keyboard controller.
  rtsp_test.py             Stream viewer.
  jetson_test_ptz_sim.py   Scripted command driver.
  from_real/dronetracker/  Read-only copy of the real tracking repo, kept as the
                           fidelity reference for the camera interface. Do not edit.

simulation/
  main_sim.py              Launches Isaac Sim and the lib_manager subprocess.
  sim_app.py               Builds the scene, camera and OmniGraph wiring.
  consts.py                Resolution, FoV, ROS topic names, RTSP port and paths.
  lib_manager.py           Runs the simulation libraries (the RTSP server).
  libraries/
    ros_image_to_rtp_lib.py  ROS image → H.264 → RTSP, plus the realism effects.
  script_nodes/            Python OmniGraph nodes (zoom, bbox, sensors).

extensions/                Isaac Sim extensions (coordinate math, position input,
                           sensor output).
usd/                       Scenes, cameras, sensors and drone assets.
```

### Handy configuration knobs

| Setting | Where | Default |
|---|---|---|
| Camera resolution | `simulation/consts.py` → `RESOLUTION_WIDTH/HEIGHT` | 1920×1080 |
| Camera FoV | `simulation/consts.py` → `CAMERA_FOV` | 78.1° |
| RTSP port / paths | `simulation/consts.py` → `RTSP_PORT`, `RTSP_PATHS` | 8554, the two camera paths |
| Pan / tilt speed | `ptz/ptz_sim.py` → `pan_vel`, `tilt_vel` | 60 °/s at full stick |
| Zoom travel time | `ptz/ptz_sim.py` → `_zoom_loop` | 7 s end to end (matches the real lens) |
| Scene start delay | env `PTZ_SCENE_DELAY_S` | 20 s after Isaac loads |
| Explicit pose target | env `PTZ_JETSON_IP` | unset (reply-to-sender) |
| RTSP diagnostics file | env `PTZ_RTSP_LOG` | `/tmp/ptz_sim_rtsp.log` |
| Show Isaac Sim's logs | `ptz/ptz_sim.py` → `show_isaac_logs` | `False` |
| Realism effects | `simulation/libraries/ros_image_to_rtp_lib.py` → `EFFECTS` | blur + flies + birds on |
