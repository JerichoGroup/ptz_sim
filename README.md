# PTZ Camera Simulation — Isaac Sim ↔ Jetson

This repo simulates the **Netz-250 PTZ security camera** so the drone-tracking algorithm can
be developed and tested without the physical camera.

From the algorithm's point of view nothing changes: it still receives an **RTSP H.264 video
stream** and still sends the **same pan / tilt / zoom commands** it would send to the real
camera. The difference is that behind the scenes those commands move a virtual camera inside
an **NVIDIA Isaac Sim** world, and the video is rendered rather than captured.

> **Which camera?** The Netz-250 (Sony IMX415 1/2.8" sensor, 20× optical). Everything the
> simulator needs to know about it lives in one file — [`ptz/netz250.py`](ptz/netz250.py) —
> mirroring the `Netz-250` entry in DroneTracker's `CAMERA_PRESETS`. The older UNV IPC6852
> is deprecated and is **not** emulated.

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
looks like `{"type": "<name>", "time": <unix-seconds>, ...}`.

These mirror the ONVIF operations the real camera receives, in the camera's own units:
**pan/tilt in ONVIF normalised `[-1,+1]`** and **zoom as the RAW ONVIF value** (`-1` = wide,
`+1` = tele on this camera).

| Command | Payload | Meaning |
|---|---|---|
| `move` | `vx`, `vy` | ONVIF *ContinuousMove*. Normalised velocity; the host slews at the datasheet rate (24.5 °/s pan, 15 °/s tilt at full stick). Expires after `cmd_ttl` (0.15 s) so the camera stops if packets stop. |
| `center_and_zoom` | `dp`, `dt`, `pan`, `tilt`, `zoom_raw`, `use_absolute` | The lock-on move. With `use_absolute` (the Netz-250 path) the Jetson has already resolved `pan = pan_now + dp`, so the host performs **one AbsoluteMove** — pan, tilt and zoom together, exactly like the camera. |
| `center_fine` | `dp`, `dt` | Fine centering via *RelativeMove*, **no zoom change**. Executed near-exactly (not snapped to the coarse grid) — this is why it exists. |
| `relative_move` | `dp`, `dt` | Plain *RelativeMove*. Used by the debug tools in `ptz/`; the production controller uses `center_fine`. |
| `absolute_move` | `pan`, `tilt`, `zoom` | *AbsoluteMove*. Any axis may be omitted to leave it untouched. Pan/tilt snap to the grid. |
| `zoom_to` | `raw` | Absolute zoom to a RAW ONVIF value. The host slews over ~4 s end-to-end. |
| `save_home` / `go_home` | – (`go_home` may carry `zoom_raw`) | Capture / return to the home pose. `go_home` carries the zoom to restore, matching the real controller's precedence (home zoom → pre-lock zoom → search position). |
| `stop_all` | – | Stop all motion. **Does not shut the simulator down** — the Jetson sends this on every quit, and Isaac Sim takes too long to start to throw it away. |
| `ping` | `scene`, `session` | Heartbeat, every 1.5 s. See below. |

### ② Pose — sim stand → Jetson (UDP, JSON, 20 Hz)

The sim replies `{"type":"pose","pan":…,"tilt":…,"zoom":…,"timestamp":…}` twenty times a
second, so the Jetson always knows where the camera actually is (it drives the HUD and keeps
the zoom value honest).

`pan`/`tilt` are ONVIF units and `zoom` is the **RAW** ONVIF value — precisely what
`GetStatus` returns on the real camera, so `PTZController.get_pose()` and
`PTZSimController.get_pose()` are interchangeable. The Jetson normalises the raw zoom to
`[0,1]` itself.

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
sprites — small fast "flies" and animated "birds" — so the detector sees the kind of
clutter it meets in the field. Everything is tunable via the `EFFECTS` dataclass at the top
of that file.

Birds spawn mostly in the **upper two thirds** of the frame (`top_frac` / `top_weight` on
`SpriteKind` — currently 85% sky / 15% ground): a bird on the ground is not a relevant
target, but the detector should still meet one occasionally.

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
  model: "Netz-250"            # must match what the simulator emulates
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

### Zoom does nothing / the image never changes field of view

Zoom crosses two processes: `ptz_sim.py` publishes `/isaac_core/zoom`, and
`simulation/script_nodes/zoom_node.py` (running **inside Isaac Sim**) turns that into a focal
length. The Isaac side logs to its own file, so check it first:

```bash
tail -f /tmp/ptz_sim_zoom.log
```

| What you see | Meaning |
|---|---|
| `zoom=0.42 mag=2.1x FL=… HFoV=…` | Working — zoom is being applied. |
| `WARNING: no messages on /isaac_core/zoom yet` | The node is alive but nothing is publishing. Is `ptz_sim.py` running? Did the Jetson send a zoom? |
| `FATAL: camera prim not found` | `CAMERA_PRIM_PATH` in `zoom_node.py` doesn't match the loaded camera USD. |
| nothing at all | The module body crashed, so `setup`/`compute` were never extracted. **Almost always cause:** something at MODULE LEVEL called a function defined in the same file — see the warning block at the top of `zoom_node.py`. `omni.graph.scriptnode` only patches the script's names into `setup`/`compute`/`cleanup` globals *after* the module body has run, so such a call raises `NameError`, the exec aborts, and the node does nothing at all — with no log, because the logging call is what died. |

Note the host also prints `[Host] Zoom target → raw …` when a zoom command arrives, so you can
tell a missing command apart from a command that arrived and wasn't applied.

### The Jetson prints `404 Not Found`

The path is wrong. The simulator mounts `/Streaming/channel/1` (the Netz-250's main path),
`/cam/realmonitor` and the legacy `/unicast/c1/s0/live`. Clear any `rtsp_main_override` /
`rtsp_alt_override` in the Jetson's `config.yaml` so the URL is built from `camera.model`.

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

## 6. Interface parity — real camera vs. simulated camera

The whole point of this simulator is that **the tracking algorithm cannot tell the
difference**. On the `DroneTracker` side that is enforced by two classes with the *same
public interface*:

| | Class | File |
|---|---|---|
| Real camera | `PTZController` | `dronetracker/ptz/controller.py` |
| Simulated camera | `PTZSimController` | `dronetracker/ptz/sim_controller.py` |

`PTZController` **defines** the interface (it is the real hardware contract);
`PTZSimController` **follows** it, replacing ONVIF/SOAP calls with JSON-over-UDP.

The pipeline never picks one itself — it accepts whichever it is handed:

```python
LivePtzPipeline(cfg, ptz=None)   # ptz=None  -> builds the real PTZController
                                 # ptz=<obj> -> uses the injected controller
```

`run_live_ptz.py` passes nothing (real camera); `sim_motion_ptz_pipeline.py` injects a
`PTZSimController`. That single optional argument is the *only* simulator-related change to
the shared pipeline code.

### Verified state (2026-07-29, Netz-250)

The two public surfaces are **identical** — same method names, same signatures, nothing on
one that is missing from the other:

```
autofocus_once()
center_and_zoom(delta_pan, delta_tilt, zoom_target,
                space=None, parallel=True, use_absolute=False) -> bool
center_fine(delta_pan, delta_tilt)                             -> bool
get_pose()                      -> (pan, tilt, RAW zoom, timestamp)
go_home()
move(vx, vy)
save_home()
stop_all()
vel_scale()                     -> float
zoom_in()   zoom_out()   zoom_to(t)
```

Plus the state the pipeline reads directly: `ready`, `connect_error`, `zoom_pos`,
`_zooming`, `_rel_moving`.

This is checked automatically, not by eye:
`tests/test_sim_controller.py::test_public_surface_matches_real_controller` reflects over
`PTZController` and asserts the sim matches it in **both** directions — a missing method
*and* an extra method both fail the test. Run it with:

```bash
cd ~/clones/DroneTracker && python3 -m pytest tests/test_sim_controller.py -v
```

> An **extra** method on the sim is treated as a failure on purpose: it lets code work in
> simulation and then break on the real camera, which is the exact bug class this simulator
> is supposed to prevent.

### Camera swap history

The project originally emulated the **UNV IPC6852** (45x, `TranslationSpaceFov` centering).
When DroneTracker moved to the **Netz-250**, the *interface* barely changed — they had
refactored camera differences into `CAMERA_PRESETS` data — but the *behaviour* changed a lot:
FoV-fraction centering became ONVIF-unit centering via measured curves, RelativeMove lock-on
became a single AbsoluteMove, zoom raw range flipped from `0..1` to `-1..+1`, ContinuousMove
signs reversed, and `center_fine()` appeared. Section 7 covers what the simulator now
reproduces. The IPC6852 is deprecated and no longer emulated.

### What was reconciled to get there

Four discrepancies were found and fixed on the `DroneTracker` side:

| Issue | Why it mattered |
|---|---|
| `zoom_to()` gating (revisited for the Netz-250) | The IPC6852 dropped a zoom requested mid-motion; the Netz-250 instead **coalesces** to the latest target, because only AbsoluteMove moves its zoom and each takes ~2 s. The sim now accumulates too, so rapid `Z`/`X` presses behave identically. |
| `center_and_zoom(dp, dt, …)` vs `(delta_pan, delta_tilt, …)` | Worked only because the caller passes those three positionally. Any keyword call would have raised on one class and not the other. |
| Sim had `relative_move()`, `zoom_search()`, `zoom_track()`, `pid_move` | All four had been **deleted from the real controller**; none are called by the pipeline. Leaving them invited code that runs in simulation and fails on hardware. |
| Sim had `trigger_autofocus()`, real has `autofocus_once()` | Same concept under two names. Renamed to the real one (still a no-op — there is no lens to focus). |

### Behaviours that match by construction

* **Lock-on geometry.** `center_and_zoom`'s `delta_pan`/`delta_tilt` are ONVIF pan/tilt
  units. The simulator's field of view is derived from the same measured `pan_shift_curve`
  the algorithm centres with, so a commanded delta shifts the image by exactly the fraction
  the algorithm assumes — at every zoom level (see section 7).
* **Velocity scaling.** `vel_scale()` uses the identical formula on both sides
  (`max(min_vel_scale, 1 - zoom_pos·(1 - min_vel_scale))`), so arrow-key panning slows down
  with zoom the same way. *(This was a real bug: the sim used an older FoV-ratio formula
  long after the real controller switched to the linear one.)*
* **Command coalescing.** `move()` is latest-intent on both sides and expires after
  `cmd_ttl` (0.15 s), and both refuse to issue a `move` while a relative move is in flight
  (a ContinuousMove would cancel it).
* **Zoom travel time.** The simulator slews zoom over 4 s end-to-end, matching the
  Netz-250's `zoom.full_travel_s = 4.0`, and the Jetson passes that value through so the two
  cannot drift apart.
* **Zoom coalescing.** Rapid zoom presses accumulate into one final move on both sides — the
  camera only honours AbsoluteMove for zoom, so dropping requests would feel different.
* **Raw zoom range.** Pose carries the RAW ONVIF zoom (`-1 … +1`); both sides normalise it to
  `[0,1]` with identical formulas.
* **Home.** The sim auto-saves home at the startup pose, mirroring the real controller's
  connect-time home capture, so `go_home` works from the first frame.

### Known asymmetries (deliberate, and safe)

These exist because the transports genuinely differ. None are visible to the pipeline —
verified by grepping every `_ptz.*` access in `pipeline/`, `apps/` and `rendering/`.

| Only on the real controller | Why it is fine |
|---|---|
| `ptz`, `token` | ONVIF service handles — meaningless without a camera. |
| `home_pos`, `home_zoom`, `kp_home`, `home_tol`, `home_timeout`, `home_max_vel` | Parameters for the real go-home control loop. The simulator slews home itself with a cosine ease, so the Jetson does not need them. |
| `zoom_full_s` | See the caveat below. |
| `_homing` | Internal to the real mover thread's Stop-suppression logic. |

| Only on the sim controller | Why it is fine |
|---|---|
| `sock`, `host_addr`, `listen_port`, `pose` | UDP transport internals, the analogue of `ptz`/`token`. |

Constructor arguments differ too, by necessity — the real one takes `ip/port/user/passwd`,
the sim takes `host_ip/host_port/listen_port` plus `scene`. Nothing constructs these
directly except `run_live_ptz.py` and `apps/sim_live_ptz.py`.

> ✅ `zoom.full_travel_s` is now passed to `PTZSimController` and used for its settle maths,
> and the host's slew rate comes from `netz250.ZOOM_FULL_TRAVEL_S`. Keep those two in step if
> you ever change the camera's zoom speed.

---

## 7. What makes this a *Netz-250* and not just "a camera"

Everything below lives in [`ptz/netz250.py`](ptz/netz250.py). The point of reproducing these
details — including the awkward ones — is that the algorithm was calibrated against the real
hardware. A simulator that behaved *better* than the camera would hide real bugs.

### Field of view comes from a measured curve, not the datasheet

The camera's on-screen display claims up to 30× magnification. That number is wrong — it
includes digital zoom and simply overstates the optics. DroneTracker measured the truth by
panning a known ONVIF delta and watching how far the scene actually moved
(`scripts/calib_zoom_via_pan.py`), giving `pan_shift_curve`: the fraction of the frame the
scene shifts per ONVIF unit, at each zoom level.

| zoom | true optical mag | on-screen claim | HFoV |
|---|---|---|---|
| 0.0 | 1.0× | 1× | 65.0° |
| 0.2 | 1.4× | 4× | 45.9° |
| 0.5 | 3.0× | 10× | 21.7° |

So the simulator derives its field of view from that same curve:

```
HFoV(Z) = (pan_range_deg / 2) / shift_per_pan(Z)
```

Pan/tilt then stay **linear** in ONVIF units (180°/unit pan, 45°/unit tilt). This is not a
coincidence — it makes the geometry cancel exactly. The algorithm centres a target with
`dx = fov_gain · ex / shift_per_pan(Z)`; the resulting image shift in the simulator is

```
dx · 180 / HFoV(Z)  =  (fov_gain · ex / spp(Z)) · spp(Z)  =  fov_gain · ex
```

— independent of zoom. Verified: a target 25 % off-centre produces a 0.205 frame shift at
every zoom level, which is exactly `fov_gain (0.82) × 0.25`. Had we rendered the on-screen
magnification instead, lock-on would overshoot by 3–5× at zoom.

> **Known gap:** the measured curves stop at zoom **0.5**. Above that the simulator continues
> the trend using the *shape* of the on-screen curve, anchored to the last real measurement.
> Extend `PAN_SHIFT_CURVE` / `TILT_SHIFT_CURVE` with measurements above 0.5 to remove the
> guesswork. The algorithm has the same gap (its own curve lookup clamps at 0.5).

### Pan/tilt direction

The camera is ceiling-mounted (hanging upside down), and the algorithm's sign constants were
tuned against that physical install. Two rules make the whole chain consistent:

* **increasing ONVIF pan = look right**, **increasing ONVIF tilt = look up**
  (`ptz_sim.py::_update_gimbal` negates both, because Isaac's ENU gimbal is the opposite
  handedness: `+yaw` = CCW = left, `+pitch` = nose down);
* `PTZSimController.move()` multiplies velocity by `continuous_pan_sign` / `continuous_tilt_sign`
  (both `-1` here) before sending — exactly as `PTZController._mover` does, so the simulator
  receives the same camera-frame velocity the real camera would.

Both halves matter. Getting only one right is what produced the earlier symptom where the
**arrow keys worked but lock-on moved away from the target** — the manual path went through two
sign flips and cancelled out, while the centering path went through one.

If a lock-on ever moves the wrong way, flip `frozen.fov_sign_x` / `fov_sign_y` in the Jetson's
`config.yaml` — no code change needed.

### The AbsoluteMove grid quirk — reproduced on purpose

On the real camera, **AbsoluteMove snaps pan/tilt onto a ~0.02 ONVIF grid** (~3.6° of pan).
Small deltas — exactly the ones needed to centre at high zoom — either round away to nothing
or overshoot. That single quirk is the reason `PTZController.center_fine()` exists, and it is
faithfully reproduced:

```
AbsoluteMove(0.005) -> quantised to 0.0000   (the move is lost)
center_fine(0.005)  -> lands at    0.0050   (RelativeMove is exact)
```

### Other behaviours that are deliberately camera-accurate

| Behaviour | Value | Why it matters |
|---|---|---|
| Zoom RAW range | `-1 … +1` | Raw `0.0` is **mid**-zoom, not wide. Treating it as `[0,1]` would silently halve every magnification. |
| Zoom travel | ~4 s end-to-end | Lock-on settle timing depends on it. |
| Zoom commands | latest-target *chase* | The camera only honours AbsoluteMove for zoom (~2 s each), so rapid Z/X presses **accumulate** into one final move rather than being dropped. |
| ContinuousMove signs | pan −1, tilt −1 | A positive `vx` makes the reported pan **decrease** on this camera. |
| Slew speeds | 24.5 °/s pan, 15 °/s tilt | Datasheet, at full stick. |
| Tilt range | 90° total | Tilt is mechanically limited; both axes clamp at ±1. |
| Stream | 1920×1080 @ 20 fps | Matches the camera; the RTSP caps advertise 20 fps. |
| RTSP path | `/Streaming/channel/1` | Hikvision-style, unlike the old UNV path. All three paths stay mounted. |

---

## 8. Where things live

```
ptz/
  ptz_sim.py               Host entry point. UDP command server + gimbal/zoom
                           publishing + pose replies + scene playback.
  netz250.py               THE CAMERA MODEL — lens curves, ranges, quirks.
                           Start here when something about the camera is wrong.
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
  script_nodes/
    zoom_node.py           Maps /isaac_core/zoom to the camera's focal length using
                           the SAME measured curve as netz250.py (keep in sync).
    bbox_node.py, sat_node.py, sensor_node.py

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
