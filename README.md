# PTZ Camera Simulation — Isaac Sim ↔ Jetson

This repo simulates the **Netz-250 PTZ security camera** so the drone-tracking algorithm can
be developed and tested without the physical camera — and without a real drone to point it at.

From the algorithm's point of view nothing changes. It still receives an **RTSP H.264 video
stream** and still sends the **same pan / tilt / zoom commands** it would send to the real
camera. Behind the scenes those commands move a virtual camera inside an **NVIDIA Isaac Sim**
world, the video is rendered rather than captured, and a simulated target drone flies through
it for the tracker to find.

> **Which camera?** The Netz-250 (Sony IMX415 1/2.8" sensor). Everything the simulator knows
> about it lives in one file — [`ptz/netz250.py`](ptz/netz250.py) — mirroring the `Netz-250`
> entry in DroneTracker's `CAMERA_PRESETS`. The older UNV IPC6852 is deprecated and is **not**
> emulated.

Two machines, two repos:

| Machine | Nickname | Runs | Repo |
|---|---|---|---|
| Workstation with an NVIDIA GPU | **the sim stand** | Isaac Sim, the PTZ command server, the RTSP video server, the target drone | `ptz_sim` (this repo) |
| NVIDIA Jetson | **the Jetson** | The real tracking pipeline — motion detection, YOLO, PTZ control, GUI | `DroneTracker`, branch `feature/ptz_sim` |

**Reading order if you are new:** §1 to understand the shape, §2–§3 to get it running, §4 for
the controls. §5 onward is reference.

---

## 1. How it fits together

Four independent flows. Three cross the machine boundary; one is internal to the sim stand.

```
                    SIM STAND                                          JETSON
        ┌──────────────────────────────────────────┐          ┌──────────────────────────┐
        │   ptz/ptz_sim.py   (PTZSim)              │          │  sim_motion_ptz_pipeline │
        │   binds UDP :5005  ◄─────────────────────┼──① cmds──┤  PTZSimController        │
        │                    ──────────────────────┼──② pose──►  binds UDP :5006         │
        │          │                               │          │            │             │
        │          │ ROS 2 topics                  │          │            │ frames      │
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
        │   ptz/drone.py (SimDrone)                │          │                          │
        └──────────────────────────────────────────┘          └──────────────────────────┘
```

**① Commands (Jetson → sim, UDP 5005).** One small JSON packet per command,
`{"type": "<name>", "time": <unix-seconds>, ...}`, mirroring the ONVIF operations the real
camera receives in the camera's own units: **pan/tilt in ONVIF `[-1,+1]`** and **zoom as the
RAW ONVIF value** (`-1` = wide, `+1` = tele on this camera).

| Command | Payload | Meaning |
|---|---|---|
| `move` | `vx`, `vy` | ONVIF *ContinuousMove*. The host slews at 24.5 °/s pan, 15 °/s tilt at full stick, and stops if packets stop arriving (0.15 s TTL). |
| `center_and_zoom` | `dp`, `dt`, `pan`, `tilt`, `zoom_raw`, `use_absolute` | The lock-on move. On the Netz-250 path the Jetson has already resolved absolute targets, so the host does **one AbsoluteMove** — pan, tilt and zoom together, like the camera. |
| `center_fine` | `dp`, `dt` | Fine centering via *RelativeMove*, no zoom change. Exact, not snapped to the coarse grid — which is the whole reason it exists (see §8). |
| `absolute_move` | `pan`, `tilt`, `zoom` | *AbsoluteMove*; any axis may be omitted. Pan/tilt snap to the grid. |
| `zoom_to` | `raw` | Absolute zoom to a RAW value; slews over ~4 s end to end. |
| `save_home` / `go_home` | – / `zoom_raw` | Capture / return to the home pose. |
| `stop_all` | – | Stop motion. **Does not shut the simulator down** — the Jetson sends this on quit and Isaac takes too long to restart. |
| `ping` | `session` | Heartbeat every 1.5 s. Keeps the pose return path alive (see below). |
| `scene` / `drone_vel` / `drone_stop` | see §4 | Simulator-only: drive the target drone. |
| `relative_move` | `dp`, `dt` | Plain *RelativeMove*, used by this repo's debug tools only. |

**② Pose (sim → Jetson, UDP, 20 Hz).** `{"type":"pose","pan":…,"tilt":…,"zoom":…,"timestamp":…}`,
so the Jetson always knows where the camera actually is. `zoom` is the **RAW** value —
precisely what `GetStatus` returns on the real camera — and the Jetson normalises it itself.
Simulator-only extras ride along on the same packet: `drone{fwd,right,up,range}`, `scene`,
`joystick`.

The key design point: **the sim never initiates traffic.** It replies to the source address of
whatever packet it last received, from the same socket the commands arrived on. So there is
**no Jetson IP to configure anywhere** — it works through NAT, a VPN or a plain USB link — and
the 1.5 s heartbeat matters even when nobody touches the controls, because it keeps that
return path fresh.

**③ Video (sim → Jetson, RTSP/H.264 over TCP).** Isaac publishes rendered 1920×1080 frames on
`/isaac_core/image_rgb`; `simulation/libraries/ros_image_to_rtp_lib.py` encodes them and serves
RTSP on **8554** at all three paths the camera family uses, so the Jetson needs no
simulator-specific URL:

```
/Streaming/channel/1     ← the Netz-250's main path
/cam/realmonitor         ← the fallback
/unicast/c1/s0/live      ← legacy UNV path, still mounted for old tools
```

A real camera serves RTSP on privileged port **554**, which a normal user cannot bind — hence
one `iptables` redirect in §2. **The Jetson opens this connection**, so the return path is
never an issue.

Before encoding, a small **realism layer** is applied: a Gaussian refocus blur that ramps up
and clears after each zoom change (the real lens hunting focus), plus drifting sprites — small
fast "flies" and animated "birds", spawned mostly in the upper two thirds of frame — so the
detector meets field-like clutter. All tunable via `EFFECTS` at the top of that file.

**④ Target drone (internal, UDP 33335).** `ptz/drone.py` flies a simulated drone inside Isaac,
driven live from the Jetson's keyboard. Covered in §4.

### Ports

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 5005 | UDP | Jetson → sim | PTZ + drone commands |
| 5006 | UDP | sim → Jetson | Pose replies (reply-to-sender) |
| 554 → 8554 | TCP | Jetson → sim | RTSP video (554 redirected to 8554) |
| 33335 | UDP | internal to sim | Target drone pose → Isaac Sim |

### The two repos

The algorithm cannot tell the simulator from the real camera because both are driven through
the *same* interface:

| | Class | File |
|---|---|---|
| Real camera | `PTZController` | `DroneTracker/dronetracker/ptz/controller.py` |
| Simulated camera | `PTZSimController` | `DroneTracker/dronetracker/ptz/sim_controller.py` |

`PTZController` **defines** the contract (it is the hardware); `PTZSimController` **follows**
it, swapping ONVIF/SOAP for JSON-over-UDP. The pipeline never chooses — it accepts whichever
it is handed, and that single optional argument is the only simulator-related change to shared
pipeline code:

```python
LivePtzPipeline(cfg, ptz=None, sim=None)   # ptz=None -> the real PTZController
                                           # ptz=<obj> -> the injected controller
```

`run_live_ptz.py` passes nothing (real camera). `sim_motion_ptz_pipeline.py` injects the sim
controller plus `SimControls` (the simulator-only GUI half: scene keys, manual flight, range
readout).

The two public surfaces are kept **identical** by a test that reflects over `PTZController` and
fails in *both* directions — a missing method and an **extra** method are both failures, since
an extra one lets code pass in simulation and break on hardware:

```bash
cd ~/clones/DroneTracker && python3 -m pytest tests/test_sim_controller.py -v
```

Anything simulator-only belongs in `SimControls`, never on the controller.

---

## 2. One-time setup

### On the sim stand

**Redirect the RTSP port** so the Jetson can use the standard camera URL. Once **per boot** —
it does not survive a reboot:

```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 554 -j REDIRECT --to-port 8554
sudo iptables -t nat -L PREROUTING -n --line-numbers | grep 8554    # confirm
```

<details>
<summary>Prefer not to touch iptables?</summary>

Point the Jetson straight at 8554 instead, though its config is then no longer identical to
the real-camera one:

```yaml
camera:
  rtsp_main_override: "rtsp://<SIM_IP>:8554/Streaming/channel/1"
  rtsp_alt_override:  "rtsp://<SIM_IP>:8554/cam/realmonitor"
```
</details>

**Install the Python packages** (once):

```bash
pip install ./simulation      # installs isaac_core_dev_kit
pip install ./debugger        # ros-sender / udp-sender GUI tools
```

`ISAACSIM_PYTHON` must be on your PATH — it is a shell alias for Isaac's bundled `python.sh`.

### On the Jetson

```bash
cd ~/clones/DroneTracker
git checkout feature/ptz_sim
./setup_jetson_env.sh          # add --yes to skip prompts
```

Use this script rather than `pip install torch`: it installs the **JetPack-matched** PyTorch. A
plain wheel reports a version but cannot see the GPU.

Then check `config.yaml` — see §6, and in particular **verify the IP addresses**, which is the
single most common way to lose an afternoon.

> A TensorRT `.engine` only loads on the exact TensorRT version that built it, so build engines
> **on the Jetson**: `python scripts/export_engine.py`.

---

## 3. Running it

Two terminals on two machines. **Start the sim stand first** — Isaac Sim takes ~15 s to load.

### Step 1 — sim stand

```bash
cd ~/clones/ptz_sim
ISAACSIM_PYTHON ./ptz/ptz_sim.py
```

Wait for these lines before touching the Jetson:

```
[Host] PTZSim listening on 0.0.0.0:5005
[Host] Jetson pose target: reply-to-sender (source of received commands)
[Host] scenes OK: [1, 2]
IsaacSim loaded in 14.02 seconds!
[Host] Isaac Sim started.
[Host] Home saved: pan=0.00, tilt=0.00, zoom=0.00
```

If `scenes OK` is replaced by warnings, a scene in `ptz/scenes.py` has a mistake — see §4.

The RTSP server runs in a subprocess whose output is hidden, so it logs to a file. Watching it
in a second terminal is the single most useful thing when video misbehaves:

```bash
tail -f /tmp/ptz_sim_rtsp.log
```

Healthy output (`image_topic` = frames from Isaac, `pushed` = frames to a connected client):

```
[RTSP] Mounted at /Streaming/channel/1
[RTSP] Server listening on :8554
[RTSP] no client: image_topic=20.0 fps  pushed=0.0 fps
[RTSP] Client connected — pipeline started
[RTSP] client attached: image_topic=20.0 fps  pushed=20.0 fps
```

### Step 2 — Jetson

```bash
cd ~/clones/DroneTracker
python3 ./sim_motion_ptz_pipeline.py
```

A window opens with the simulated camera view and the tracker runs against it exactly as it
would against the real camera.

> Needs a real display. Over SSH, prefix with `DISPLAY=:0` to put the window on the Jetson's
> attached monitor.

### Is it working?

Within a second or two of the Jetson starting, the **sim stand** should print the heartbeat:

```
[Host] From ('192.168.55.1', 5006): {'type': 'ping', 'session': '…'}
```

That one line proves commands are crossing. On the Jetson you should see `[Cam] 1920x1080 @
20fps` and a `[Diag]` line each second.

### Nothing responds? Check the IP first

If the Jetson logs its own actions (`[SIM] manual flight ON`, camera moves in the HUD) but the
**sim stand prints nothing at all**, the commands are going to the wrong address. Confirm the
sim stand's current address and that the Jetson's config matches:

```bash
# on the sim stand
ip -4 -o addr show | awk '{print $2, $4}'
```

```bash
# on the Jetson
grep -n "host_ip\|^  ip:" config.yaml
```

Both `camera.ip` and `sim.host_ip` must point at an address the sim stand **currently holds**.
VPN/PPP addresses are handed out dynamically and change on reconnect, so prefer the stable
direct link (typically the Jetson is `192.168.55.1` and the sim stand `192.168.55.100`).

This failure is nastier than it sounds: a stale IP can leave **video working while commands
vanish**, so the system looks half-alive and you chase the wrong thing. If in doubt, check the
IP before anything else.

### Shutting down

Quit the Jetson with `q` first, then `Ctrl-C` the sim stand. `stop_all` deliberately does not
stop the simulator, so you can restart the Jetson side as often as you like while Isaac stays
up.

### Other entry points

| Where | Command | What it does |
|---|---|---|
| Jetson | `python3 ./run_live_ptz.py` | The **real camera** pipeline, unchanged by any of this. |
| Sim stand | `python3 ptz/rtsp_test.py` | View the stream alone, to isolate video from the tracker. `--backend ffmpeg` uses the Jetson's decoder. |
| Sim stand | `python3 ptz/ptz_tui.py --host <SIM_IP>` | Drive the camera by keyboard without the tracker. |
| Sim stand | `python3 ptz/jetson_test_ptz_sim.py` | Fire a scripted PTZ sequence. |

---

## 4. Controls, the target drone, and scenes

All keys work **in the pipeline video window** — click it first, or your keystrokes go to the
terminal and nothing happens.

### Camera

| Key | Action |
|---|---|
| `t` / `Enter` | Lock on to the selected target (IDLE only) |
| `r` | Return to home pose |
| `h` | Save the current pose as home |
| Arrow keys | Pan / tilt (IDLE only) |
| `z` / `x` | Zoom in / out |
| `v` | Start / stop recording |
| `s` | Save a screenshot |
| `q` | Quit |

### Target drone

The target does not exist until you ask for one. The first scene or manual-flight keypress
spawns it **200 m in front of the camera and 40 m up** (`SPAWN_OFFSET` in `ptz/scenes.py`) —
spawning it *at* the camera used to black out the video.

| Key | Action |
|---|---|
| `1` … `9` | Play that scene. Cancels whatever the target was doing. |
| `0` | Toggle manual flight on/off |
| `i` / `k` | Forward (away from the camera) / backward (toward it) |
| `j` / `l` | Left / right |
| `u` / `o` | Up / down |

Manual flight is **camera-relative** and runs at `PTZ_DRONE_SPEED_MS` (default 10 m/s). A
single press always travels at least `MIN_PRESS_TRAVEL_M` (15 m), so one tap is visible even at
several hundred metres; holding a key flies continuously. The flight letters act **only** while
manual flight is on, so a stray keypress during tracking cannot move the target.

The HUD shows the live camera-to-target offset, closed-loop from Isaac's own world poses:

```
RANGE  : 266 m   fwd -150  right +200  up +90
```

There is deliberately **no in-frame indicator** — Isaac's 2D-bbox visibility is unreliable for
this target and reported "not in frame" even when the drone was plainly centred. A readout that
is always wrong is worse than none.

### Editing and adding scenes

Scenes live in the `SCENES` dict in [`ptz/scenes.py`](ptz/scenes.py) and are **declarative** —
no Python knowledge needed. Each step is one block:

```python
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
}
```

`meters` is how far to travel, `seconds` is how long to take (so speed is implied — 400 m in
8 s is 50 m/s). Directions are **relative to the camera**, exactly like the manual flight keys.

| `move` | Meaning | Needs |
|---|---|---|
| `forward` / `backward` | Away from / toward the camera | `meters`, `seconds` |
| `left` / `right` | Sideways | `meters`, `seconds` |
| `up` / `down` | Altitude | `meters`, `seconds` |
| `wait` | Hover in place | `seconds` |
| `goto` | Fly to an absolute coordinate | `lat`, `lon`, `alt`, `seconds` |

To add one, copy a block and give it an unused number 1–9. Keys `1`–`9` map straight to these
numbers, so scene `3` is played by pressing `3`.

Mistakes are caught at **sim-stand startup** and reported by number and step, so you find out
before you fly:

```
[Host] WARNING: 1 problem(s) in ptz/scenes.py — those scenes will not fly correctly:
[Host]   scene 3 step 2 (forward): 'meters' must be a number, got 'abc'
```

The camera itself sits at `CAMERA` in the same file (lat 32.20647, lon 35.29034, alt 540,
looking east at yaw 90°). Change it and the whole scene geometry follows, because everything is
camera-relative.

---

## 5. Troubleshooting

### The sim stand prints nothing when the Jetson sends commands

The IP is stale. See [§3 — Nothing responds](#nothing-responds-check-the-ip-first). This is the
first thing to check, always.

### The Jetson prints `DESCRIBE failed: 503 Service Unavailable`

The RTSP server is reachable but cannot serve video. Check `/tmp/ptz_sim_rtsp.log`:

* **`FATAL: could not bind RTSP port 8554`** — a **stale `lib_manager` from a previous run is
  still holding the port**. It is a subprocess of Isaac Sim, and if Isaac is killed rather than
  exited cleanly it can be left behind. Your new server then serves nobody while the dead one
  answers the Jetson with 503.

  ```bash
  ss -ltnp | grep 8554                              # see who holds it
  pkill -9 -f 'ptz_sim/simulation/lib_manager'      # clear it
  ```

  `main_sim.py` now asks the kernel to kill `lib_manager` when Isaac dies, so this should not
  recur — but it is still the first thing to check.

* **`WARNING: no frames from the image topic`** — the server is up but Isaac is not delivering
  frames. Confirm with `ros2 topic hz /isaac_core/image_rgb`.

* **`image_topic=20 fps` but `no client` forever** — frames are fine and the Jetson never
  reached *this* server. Almost always the stale-port case above.

### Zoom does nothing / the field of view never changes

Zoom crosses two processes: `ptz_sim.py` publishes `/isaac_core/zoom`, and
`simulation/script_nodes/zoom_node.py` (running **inside** Isaac) turns that into a focal
length. The Isaac side logs to its own file:

```bash
tail -f /tmp/ptz_sim_zoom.log
```

| What you see | Meaning |
|---|---|
| `zoom=0.42 mag=2.1x FL=… HFoV=…` | Working. |
| `WARNING: no messages on /isaac_core/zoom yet` | Node alive, nothing publishing. Is `ptz_sim.py` running? Did the Jetson send a zoom? |
| `FATAL: camera prim not found` | `CAMERA_PRIM_PATH` does not match the loaded camera USD. |
| **nothing at all** | The module body crashed, so `setup`/`compute` were never extracted. **Almost always:** something at MODULE LEVEL called a function defined in the same file. `omni.graph.scriptnode` only patches the script's names into the node's globals *after* the body has run, so such a call raises `NameError`, the exec aborts, and the node does nothing — silently, because the logging call is what died. See the warning block at the top of `zoom_node.py`. |

The host also prints `[Host] Zoom target → raw …` when a zoom arrives, so you can tell a
missing command from one that arrived and was not applied.

### Other symptoms

| Symptom | Cause and fix |
|---|---|
| `404 Not Found` | Wrong path. Clear any `rtsp_main_override` / `rtsp_alt_override` so the URL is built from `camera.model`. |
| `Connection refused` on port 554 | The `iptables` redirect is missing — it is lost on every reboot. See §2. |
| Commands work but no video (or vice versa) | They are independent flows: UDP 5005 vs TCP 554/8554. A working scene proves only the command path. |
| `CUDA initialization failure` on the Jetson | A non-Jetson PyTorch wheel. Re-run `./setup_jetson_env.sh`; a correct install reports `+cu126`. |
| `[Storage] ERROR: could not create chunk dir: Permission denied` | Recording only; the tracker still runs. `sudo chown -R $USER /mnt/ssd/dronetracker`. |
| Cesium tile / `curl: Couldn't connect` errors | Terrain tile server, unrelated to PTZ or video. Ignore. |
| Keys do nothing | The terminal has focus, not the video window. Click the window. |

---

## 6. Things you may need to adjust

### The Jetson's `config.yaml`

```yaml
camera:
  model: "Netz-250"            # must match what the simulator emulates
  ip: "<SIM_STAND_IP>"         # the only real/sim difference
sim:
  host_ip: "<SIM_STAND_IP>"    # MUST match an address the sim stand holds now
  host_port:   5005
  listen_port: 5006
yolo:
  model_path: "models/yolo26n.engine"   # must exist, built on this Jetson
```

⚠️ **`camera.ip` and `sim.host_ip` are the two values most likely to bite you.** They are
independent, so a stale one can break commands while video keeps working. Re-check both
whenever the sim stand reconnects to a VPN or changes network. There is no `sim.scene` key —
scenes are keyboard-driven now.

### Sim-stand knobs

| Setting | Where | Default |
|---|---|---|
| Camera resolution | `simulation/consts.py` → `RESOLUTION_WIDTH/HEIGHT` | 1920×1080 |
| Stream frame rate | `simulation/consts.py` → `CAMERA_FPS` | 20 fps (camera's native rate) |
| Camera FoV | `simulation/consts.py` → `CAMERA_FOV` | 64.98° (Netz-250 wide end) |
| RTSP port / paths | `simulation/consts.py` → `RTSP_PORT`, `RTSP_PATHS` | 8554, the three camera paths |
| Pan / tilt range | `ptz/netz250.py` → `PAN_RANGE_DEG`, `TILT_RANGE_DEG` | 360° pan, 90° tilt (180°/unit, 45°/unit) |
| Pan / tilt speed | `ptz/netz250.py` → `PAN_SPEED_DEG_S`, `TILT_SPEED_DEG_S` | 24.5 °/s, 15 °/s at full stick |
| Zoom travel time | `ptz/netz250.py` → `ZOOM_FULL_TRAVEL_S` | 4 s end to end (measured) |
| Camera position | `ptz/scenes.py` → `CAMERA` | 32.20647, 35.29034, 540 m, yaw 90° |
| Target spawn point | `ptz/scenes.py` → `SPAWN_OFFSET` | 200 m forward, 40 m up |
| Target drone speed | env `PTZ_DRONE_SPEED_MS` | 10 m/s while a key is held |
| Travel per keypress | `ptz/drone.py` → `MIN_PRESS_TRAVEL_M` | 15 m minimum |
| Explicit pose target | env `PTZ_JETSON_IP` | unset (reply-to-sender) |
| RTSP diagnostics file | env `PTZ_RTSP_LOG` | `/tmp/ptz_sim_rtsp.log` |
| Zoom diagnostics file | `simulation/script_nodes/zoom_node.py` → `ZOOM_LOG_PATH` | `/tmp/ptz_sim_zoom.log` |
| Show Isaac Sim's logs | `ptz/ptz_sim.py` → `show_isaac_logs` | `False` |
| Realism effects | `simulation/libraries/ros_image_to_rtp_lib.py` → `EFFECTS` | blur + flies + birds on |

---

## 7. Why this behaves like a real Netz-250

The algorithm was calibrated against real hardware, so a simulator that behaved *better* than
the camera would hide real bugs. These details are reproduced on purpose. All of them live in
[`ptz/netz250.py`](ptz/netz250.py) — start there when something about the camera looks wrong.

### Field of view comes from a measured curve, not the datasheet

The camera's on-screen display claims up to 30× magnification. That number is wrong — it
includes digital zoom and overstates the optics. DroneTracker measured the truth by panning a
known ONVIF delta and watching how far the scene actually moved
(`scripts/calib_zoom_via_pan.py`), producing `PAN_SHIFT_CURVE`: the fraction of frame the scene
shifts per ONVIF unit at each zoom.

| zoom | true optical mag | on-screen claim | HFoV |
|---|---|---|---|
| 0.0 | 1.00× | 1× | 64.98° |
| 0.2 | 1.42× | 4× | 45.92° |
| 0.5 | 3.00× | 10× | 21.66° |
| 1.0 | 9.00× | 30× | 7.22° |

The simulator derives its field of view from that same curve, `HFoV(Z) = (pan_range/2) /
shift_per_pan(Z)`, and keeps pan/tilt **linear** in ONVIF units. That is not a coincidence — it
makes the geometry cancel exactly. The algorithm centres with `dx = fov_gain · ex /
shift_per_pan(Z)`, so the resulting image shift is `fov_gain · ex`, *independent of zoom*.
Verified: a target 25 % off-centre produces a 0.205 frame shift at every zoom level, which is
exactly `fov_gain (0.82) × 0.25`. Had we rendered the on-screen magnification instead, lock-on
would overshoot by 3–5× at zoom.

> **Known gap:** the measured curves stop at zoom **0.5**. Above that the simulator continues
> the trend using the *shape* of the on-screen curve, anchored to the last real measurement.
> Extend `PAN_SHIFT_CURVE` with measurements above 0.5 to remove the guesswork. The algorithm
> has the same gap — its own lookup clamps at 0.5.

### The AbsoluteMove grid quirk

On the real camera, **AbsoluteMove snaps pan/tilt onto a ~0.02 ONVIF grid** (~3.6° of pan).
Small deltas — exactly the ones needed to centre at high zoom — round away to nothing or
overshoot. That one quirk is why `center_fine()` exists, and it is reproduced faithfully:

```
AbsoluteMove(0.005) -> quantised to 0.0000   (the move is lost)
center_fine(0.005)  -> lands at    0.0050   (RelativeMove is exact)
```

### Pan/tilt direction

The camera is ceiling-mounted and the algorithm's sign constants were tuned against that
install. Two rules keep the chain consistent: **increasing pan = look right, increasing tilt =
look up** (`_update_gimbal` negates both, because Isaac's ENU gimbal has the opposite
handedness), and `PTZSimController.move()` applies `continuous_pan_sign` / `continuous_tilt_sign`
(both `-1`) before sending, exactly as the real `PTZController._mover` does.

Both halves matter. Getting only one right produced a memorable symptom: **arrow keys worked
but lock-on moved away from the target**, because the manual path went through two sign flips
and cancelled while centering went through one. If a lock-on ever moves the wrong way, flip
`frozen.fov_sign_x` / `fov_sign_y` in the Jetson's config — no code change needed.

### Other deliberately camera-accurate behaviours

| Behaviour | Value | Why it matters |
|---|---|---|
| Zoom RAW range | `-1 … +1` | Raw `0.0` is **mid**-zoom, not wide. Treating it as `[0,1]` would silently halve every magnification. |
| Zoom travel | ~4 s end to end | Lock-on settle timing depends on it. |
| Zoom commands | latest-target *chase* | The camera only honours AbsoluteMove for zoom (~2 s each), so rapid `z`/`x` presses **accumulate** into one final move rather than being dropped. |
| ContinuousMove signs | pan −1, tilt −1 | A positive `vx` makes reported pan **decrease** on this camera. |
| Command coalescing | 0.15 s TTL | `move` is latest-intent and expires, so the camera stops if the network does. |
| Velocity scaling | linear in zoom | `vel_scale()` uses the identical formula on both sides, so arrow panning slows with zoom the same way. |
| Tilt range | 90° total | Mechanically limited; both axes clamp at ±1. |
| Stream | 1920×1080 @ 20 fps | Matches the camera; the RTSP caps advertise 20 fps. |
| Home | auto-saved at startup pose | Mirrors the real controller's connect-time capture, so `go_home` works from the first frame. |

Two copies of this camera model exist: `ptz/netz250.py` and `simulation/script_nodes/zoom_node.py`.
The script node is exec'd standalone inside Isaac and **cannot import from the repo**, so it
carries its own copy. **Keep them in sync** — they are verified numerically identical.

---

## 8. Where things live

```
ptz/
  ptz_sim.py               Host entry point. UDP command server, gimbal/zoom publishing,
                           pose replies, target-drone telemetry.
  netz250.py               THE CAMERA MODEL — lens curves, ranges, speeds, quirks.
  drone.py                 The simulated target drone (SimDrone): scenes + manual flight.
  scenes.py                Scene registry, camera position, spawn offset, validate().
  ptz_sim_controller.py    Stand-alone copy of the Jetson-side controller, used ONLY by the
                           debug tools below. The pipeline uses DroneTracker's own
                           dronetracker/ptz/sim_controller.py.
  ptz_tui.py               Keyboard camera controller.
  rtsp_test.py             Stream viewer.
  jetson_test_ptz_sim.py   Scripted command driver.
  from_real/dronetracker/  Read-only copy of the tracking repo, kept as the fidelity
                           reference for the camera interface. Do not edit.

simulation/
  main_sim.py              Launches Isaac Sim and the lib_manager subprocess.
  sim_app.py               Builds the scene, camera and OmniGraph wiring.
  consts.py                Resolution, fps, FoV, ROS topic names, RTSP port and paths.
  lib_manager.py           Runs the simulation libraries (the RTSP server).
  libraries/
    ros_image_to_rtp_lib.py  ROS image → H.264 → RTSP, plus the realism effects.
  script_nodes/
    zoom_node.py           Maps /isaac_core/zoom to focal length using the SAME measured
                           curve as netz250.py (keep in sync).
    bbox_node.py           Publishes target positions — the source of the range readout.
    sat_node.py, sensor_node.py

extensions/                Isaac Sim extensions (coordinate math, position input, sensors).
usd/                       Scenes, cameras, sensors and drone assets.
```

On the **DroneTracker** side, the simulator touches very little:

```
sim_motion_ptz_pipeline.py            Sim entry point. Builds the sim controller +
                                      SimControls and injects them into the pipeline.
dronetracker/ptz/sim_controller.py    PTZSimController — the UDP twin of PTZController.
dronetracker/sim/sim_controls.py      SimControls — scene keys, manual flight, telemetry.
dronetracker/apps/sim_live_ptz.py     Builds PTZSimController from config.
```

Everything else — detection, tracking, lock-on, the display loop — is shared with the real
camera path and must stay that way.
