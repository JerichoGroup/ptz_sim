"""ONVIF PTZ controller for the UNV IPC6852ER-X45 (or compatible cameras).

Constructor kwargs mirror the CONFIG constants in the pipeline script so the
caller passes its CONFIG values explicitly::

    ptz = PTZController(ip, port, user, passwd,
                        zoom_search_pos=ZOOM_SEARCH, zoom_track_pos=ZOOM_TRACK, ...)

All PTZ commands (zoom, pan/tilt) are non-blocking: zoom runs in a daemon
thread; pan/tilt uses a coalescing mover thread that issues the latest-intent
ContinuousMove at ~25 Hz.
"""

__all__ = ["PTZController", "TRANSLATION_SPACE_FOV"]

import math
import threading
import time
import numpy as np

from dronetracker.ptz.transform import _F_WIDE, _SENSOR_W

# ONVIF relative-move space that expresses translation as a fraction of the
# current field-of-view — auto-handles zoom, no focal-length math needed.
TRANSLATION_SPACE_FOV = (
    "http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationSpaceFov"
)

# Derived from lens constants — not a user-tunable value.
_FOV_WIDE = 2 * math.degrees(math.atan(_SENSOR_W / (2 * _F_WIDE)))   # ~64°


class PTZController:
    """ONVIF PTZ interface for the UNV IPC6852ER-X45 (or compatible)."""

    def __init__(
        self, ip, port, user, passwd,
        zoom_search_pos = 0.0,
        zoom_track_pos  = 0.60,
        zoom_max        = 1.0,
        zoom_step       = 0.05,
        zoom_full_s     = 7.0,
        cmd_ttl         = 0.15,
        kp_home         = 3.0,
        home_tol        = 0.01,
        home_timeout    = 8.0,
        home_max_vel    = 0.5,
        min_vel_scale   = 0.30,
    ):
        # Zoom range params
        self.zoom_search_pos = zoom_search_pos
        self.zoom_track_pos  = zoom_track_pos
        self.zoom_max        = zoom_max
        self.zoom_step       = zoom_step
        self.zoom_full_s     = zoom_full_s
        # Command coalescing
        self.cmd_ttl = cmd_ttl
        # Return-to-home
        self.kp_home      = kp_home
        self.home_tol     = home_tol
        self.home_timeout = home_timeout
        self.home_max_vel = home_max_vel
        # Velocity scaling
        self.min_vel_scale = min_vel_scale

        # Runtime state
        self.ready         = False
        self.ptz           = None
        self.token         = None
        self.zoom_pos      = 1.0
        self._zooming      = False
        self._rel_moving   = False   # True while a RelativeMove is in flight; gates re-entry
        self._homing       = False   # True while go_home AbsoluteMove is in flight
        self._lock         = threading.Lock()
        self.connect_error = None
        self.home_pos      = None
        self.home_zoom     = None
        self._return_zoom  = None   # pre-lock zoom; set by center_and_zoom, consumed by _do_go_home

        # ONVIF ImagingPort — used to disable continuous autofocus and trigger one-shot AF.
        self._imaging = None
        self._vs_token = None

        # Background pose cache — populated by _poll_pose daemon.
        self._pose_lock    = threading.Lock()
        self._pose         = (0.0, 0.0, 0.0, 0.0)   # (pan, tilt, zoom, timestamp)
        self._pose_zoom_ok = False                    # True if camera reports Zoom in GetStatus

        # Command-coalescing slot — single latest-intent, no queue.
        self._cmd_vel  = (0.0, 0.0)
        self._cmd_t    = 0.0
        self._cmd_lock = threading.Lock()
        self._run      = True

        # go_home cancellation: bumping the generation tells a running _do_go_home
        # thread that a newer invocation has superseded it.
        self._go_home_gen = 0

        threading.Thread(target=self._connect, args=(ip, port, user, passwd),
                         daemon=True, name="ptz-connect").start()
        threading.Thread(target=self._mover,
                         daemon=True, name="ptz-mover").start()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self, ip, port, user, passwd):
        try:
            from onvif import ONVIFCamera
            cam          = ONVIFCamera(ip, port, user, passwd, adjust_time=True)
            self.ptz   = cam.create_ptz_service()
            media      = cam.create_media_service()
            self.token = media.GetProfiles()[0].token
            # Set autofocus to MANUAL — AF only fires when we explicitly call autofocus_once().
            try:
                self._imaging   = cam.create_imaging_service()
                vs              = media.GetVideoSources()
                self._vs_token  = vs[0].token
                iset            = self._imaging.GetImagingSettings({'VideoSourceToken': self._vs_token})
                if iset.Focus and iset.Focus.AutoFocusMode != "MANUAL":
                    iset.Focus.AutoFocusMode = "MANUAL"
                    self._imaging.SetImagingSettings({
                        'VideoSourceToken': self._vs_token,
                        'ImagingSettings': iset,
                    })
                    print("[PTZ] Autofocus set to MANUAL")
                else:
                    print("[PTZ] Autofocus already MANUAL")
            except Exception as ae:
                print(f"[PTZ] ImagingPort setup skipped ({ae}) — camera may still auto-focus")
                self._imaging = None
            self._zoom_burst(-1.0, 8.0)   # force to physical wide-end stop
            self.zoom_pos = self.zoom_search_pos
            try:
                with self._lock:
                    status = self.ptz.GetStatus({'ProfileToken': self.token})
                pt = status.Position.PanTilt
                self.home_pos  = (float(pt.x), float(pt.y))
                self.home_zoom = self.zoom_search_pos
                print(
                    f"[PTZ] Home auto-captured: pan={self.home_pos[0]:.4f}"
                    f"  tilt={self.home_pos[1]:.4f}  zoom={self.home_zoom:.2f}"
                )
            except Exception as he:
                print(f"[PTZ] Home auto-capture skipped ({he}); press H to save manually")
            self.ready = True
            threading.Thread(target=self._poll_pose,
                             daemon=True, name="ptz-poll-pose").start()
            print("[PTZ] Ready at minimum zoom")
        except Exception as e:
            self.connect_error = str(e)
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"[PTZ] OFFLINE — {e}")
            print("  All PTZ commands (zoom, pan, tilt) will silently do nothing.")
            print("  Common causes:")
            print("    • 'onvif-zeep' package not installed  →  pip install onvif-zeep")
            print(f"    • Camera unreachable at {ip}:{port}   →  check LAN / cable")
            print("    • Wrong credentials                  →  check user / passwd")
            print(f"{sep}\n")

    # ── Internal zoom helpers ─────────────────────────────────────────────────

    def _zoom_burst(self, direction, duration, zoom_only=False):
        """Drive the zoom motor in ``direction`` for ``duration`` seconds, then stop.

        ``zoom_only=True`` omits PanTilt from the ContinuousMove velocity so that
        a concurrent pan/tilt RelativeMove is not cancelled.  The terminating Stop
        already specifies ``PanTilt:False`` so pan/tilt motion is never stopped here.
        """
        if not self.ptz:
            return
        try:
            req = self.ptz.create_type('ContinuousMove')
            req.ProfileToken = self.token
            vel = {'Zoom': {'x': float(direction)}}
            if not zoom_only:
                vel['PanTilt'] = {'x': 0.0, 'y': 0.0}
            req.Velocity = vel
            with self._lock:
                self.ptz.ContinuousMove(req)
            time.sleep(duration)   # lock released during sleep — poller stays responsive
            with self._lock:
                self.ptz.Stop({'ProfileToken': self.token, 'PanTilt': False, 'Zoom': True})
            self.autofocus_once()
        except Exception as e:
            print(f"[PTZ] zoom_burst error: {e}")

    def _do_zoom(self, target):
        """Drive to ``target`` zoom position; runs in a daemon thread."""
        self._zooming = True
        try:
            target    = float(np.clip(target, 0.0, 1.0))
            dist      = abs(target - self.zoom_pos)
            if dist < 0.01:
                return
            direction = 1.0 if target > self.zoom_pos else -1.0
            self._zoom_burst(direction, max(0.4, dist * self.zoom_full_s))
            self.zoom_pos = target

        except Exception as e:
            print(f"[PTZ] zoom error: {e}")
            self.zoom_pos = float(np.clip(target, 0.0, 1.0))
        finally:
            self._zooming = False

    # ── Background pose poller ────────────────────────────────────────────────

    def _poll_pose(self):
        """Daemon: poll GetStatus every 50 ms, cache (pan, tilt, zoom, timestamp).

        Syncs ``zoom_pos`` from hardware truth whenever the camera reports Zoom,
        making ``vel_scale()`` and the FOV HUD accurate rather than dead-reckoned.
        """
        while self._run:
            if self.ptz:
                try:
                    with self._lock:
                        status = self.ptz.GetStatus({'ProfileToken': self.token})
                    pt   = status.Position.PanTilt
                    z    = status.Position.Zoom
                    pan  = float(pt.x)
                    tilt = float(pt.y)
                    if z is not None:
                        zoom = float(z.x)
                        ok   = True
                    else:
                        zoom = self.zoom_pos   # fall back to dead-reckoning
                        ok   = False
                    with self._pose_lock:
                        self._pose_zoom_ok = ok
                        self._pose = (pan, tilt, zoom, time.time())
                        if ok:
                            self.zoom_pos = zoom   # sync to hardware truth
                except Exception:
                    pass
            time.sleep(0.05)

    def get_pose(self):
        """Return (pan, tilt, zoom, timestamp) from the background poll cache."""
        with self._pose_lock:
            return self._pose

    # ── Public zoom API ───────────────────────────────────────────────────────

    def zoom_to(self, t):
        """Non-blocking zoom to position ``t`` ∈ [0, 1]."""
        if not self._zooming and not self._rel_moving:
            threading.Thread(target=self._do_zoom, args=(t,),
                             daemon=True, name="ptz-zoom").start()

    def zoom_in(self):     self.zoom_to(min(self.zoom_max, self.zoom_pos + self.zoom_step))
    def zoom_out(self):    self.zoom_to(max(0.0,           self.zoom_pos - self.zoom_step))

    # ── Velocity scale ────────────────────────────────────────────────────────

    def vel_scale(self) -> float:
        """Pan/tilt velocity scale factor at the current zoom level.

        Returns 1.0 at the widest position and decreases linearly to
        ``min_vel_scale`` at maximum zoom, so arrow-key panning feels fast
        when scanning and precise when zoomed in.

        The previous FOV-ratio formula collapsed to the floor constant at
        zoom_pos ≈ 0.07 (for the 45× lens), making all speeds indistinguishable.
        This linear mapping spreads the range across the full zoom travel.
        """
        return max(self.min_vel_scale, 1.0 - self.zoom_pos * (1.0 - self.min_vel_scale))

    # ── Autofocus ───────────────────────────────────────────────────────────────

    def autofocus_once(self) -> None:
        """Trigger a one-shot autofocus sweep, then return to MANUAL.

        Toggles AutoFocusMode to AUTO for 0.5s then back to MANUAL.
        Uses a 1-second lock timeout to avoid blocking the pipeline.
        No-op if the ImagingPort was not set up in _connect().
        """
        if not self._imaging:
            return
        if not self._lock.acquire(timeout=1.0):
            print("[PTZ] autofocus_once: lock timeout — skipping")
            return
        try:
            iset = self._imaging.GetImagingSettings({'VideoSourceToken': self._vs_token})
            iset.Focus.AutoFocusMode = 'AUTO'
            self._imaging.SetImagingSettings({
                'VideoSourceToken': self._vs_token,
                'ImagingSettings': iset,
            })
        except Exception as e:
            print(f"[PTZ] autofocus_once error: {e}")
            return
        finally:
            self._lock.release()
        time.sleep(0.5)
        with self._lock:
            try:
                iset.Focus.AutoFocusMode = 'MANUAL'
                self._imaging.SetImagingSettings({
                    'VideoSourceToken': self._vs_token,
                    'ImagingSettings': iset,
                })
            except Exception as e:
                print(f"[PTZ] autofocus_once restore error: {e}")

    # ── Pan/tilt command coalescing ───────────────────────────────────────────

    def move(self, vx, vy):
        """Write the latest-intent pan/tilt velocity into the coalescing slot.

        Non-blocking — the mover thread picks it up within ~40 ms.
        Each call overwrites the previous so no backlog accumulates.
        Returns early (no-op) if a RelativeMove is in flight to prevent
        ContinuousMove from cancelling it.
        """
        if self._rel_moving:
            return
        with self._cmd_lock:
            self._cmd_vel = (float(vx), float(vy))
            self._cmd_t   = time.time()

    def center_and_zoom(self, delta_pan, delta_tilt, zoom_target,
                        space=None, parallel=True):
        """Center the target (RelativeMove) and zoom in, in one combined operation.

        Both motions start in the same daemon thread.  When ``parallel=True``
        (default), the zoom ContinuousMove fires *immediately after* the
        RelativeMove ONVIF call returns so both motors move concurrently.
        The zoom burst uses ``zoom_only=True`` so it carries no PanTilt component
        and cannot cancel the in-flight RelativeMove.

        When ``parallel=False``, the zoom starts after a 0.6 s pause, giving the
        camera firmware time to complete the pan/tilt move first.  Use this if you
        observe that the zoom ContinuousMove unexpectedly aborts the pan/tilt on
        this camera firmware.

        Sets ``_zooming = True`` and ``_rel_moving = True`` for the duration so:
        - the detect thread skips optical-flow frames while the lens is moving,
        - re-entry from the TRACK loop is prevented.
        Both flags are cleared in ``finally``.
        """
        if not self.ready or self._zooming or self._rel_moving:
            return False
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0)
            self._cmd_t   = 0.0
        # Snapshot zoom_pos before the thread so settle math is stable.
        zoom_pos_now = self.zoom_pos
        self._return_zoom = zoom_pos_now   # remember pre-lock zoom for _do_go_home
        dist      = abs(zoom_target - zoom_pos_now)
        zoom_dir  = 1.0 if zoom_target > zoom_pos_now else -1.0
        zoom_dur  = max(0.4, dist * self.zoom_full_s) if dist >= 0.02 else 0.0
        # Set both flags synchronously so the detect loop sees them immediately.
        self._rel_moving = True
        self._zooming    = True
        def _go():
            try:
                if not self.ptz:
                    return
                # ── Pan/tilt RelativeMove ─────────────────────────────────────
                req = self.ptz.create_type('RelativeMove')
                req.ProfileToken = self.token
                pt = {'x': float(delta_pan), 'y': float(delta_tilt)}
                if space:
                    pt['space'] = space
                req.Translation = {'PanTilt': pt, 'Zoom': {'x': 0.0}}
                req.Speed       = {'PanTilt': {'x': 1.0, 'y': 1.0}, 'Zoom': {'x': 1.0}}
                with self._lock:
                    self.ptz.RelativeMove(req)
                # ── Zoom ContinuousMove ───────────────────────────────────────
                if not parallel:
                    time.sleep(0.6)   # let pan/tilt complete first (sequential mode)
                if zoom_dur > 0.0:
                    self._zoom_burst(zoom_dir, zoom_dur, zoom_only=True)
                    with self._pose_lock:
                        self.zoom_pos = float(np.clip(zoom_target, 0.0, 1.0))
            except Exception as e:
                print(f"[PTZ] center_and_zoom error: {e}")
            finally:
                self._rel_moving = False
                self._zooming    = False
        threading.Thread(target=_go, daemon=True, name="ptz-center-zoom").start()
        return True

    def _mover(self):
        """Single pan/tilt driver loop at ~25 Hz.

        Issues ContinuousMove from the latest-intent slot while fresh.
        Stops the camera when the slot goes stale or a zoom is in progress.
        """
        moving = False
        while self._run:
            with self._cmd_lock:
                vx, vy = self._cmd_vel
                age    = time.time() - self._cmd_t
            stale = (vx == 0.0 and vy == 0.0) or age > self.cmd_ttl
            if not self.ready or not self.ptz or self._zooming or stale:
                if moving:
                    if self._rel_moving or self._homing:
                        # A RelativeMove or go_home AbsoluteMove is in flight.
                        # The prior ContinuousMove was already cancelled by the camera
                        # when the absolute/relative move started, so issuing Stop here
                        # would cancel that in-flight move — don't do it.
                        moving = False
                    else:
                        try:
                            with self._lock:
                                self.ptz.Stop({'ProfileToken': self.token,
                                               'PanTilt': True,
                                               'Zoom': False})
                        except Exception:
                            pass
                        moving = False
            else:
                try:
                    req = self.ptz.create_type('ContinuousMove')
                    req.ProfileToken = self.token
                    req.Velocity = {'PanTilt': {'x': float(vx), 'y': float(vy)},
                                    'Zoom':    {'x': 0.0}}
                    with self._lock:
                        self.ptz.ContinuousMove(req)
                    moving = True
                except Exception as e:
                    print(f"[PTZ] mover error: {e}")
                    moving = False
            time.sleep(0.04)

    # ── Home position ─────────────────────────────────────────────────────────

    def save_home(self):
        """Capture current pan/tilt/zoom as the home position."""
        if not self.ptz or not self.ready:
            print("[PTZ] save_home: not ready")
            return
        try:
            with self._lock:
                status = self.ptz.GetStatus({'ProfileToken': self.token})
            pt = status.Position.PanTilt
            self.home_pos  = (float(pt.x), float(pt.y))
            self.home_zoom = self.zoom_pos
            print(
                f"[PTZ] Home saved: pan={self.home_pos[0]:.4f}"
                f"  tilt={self.home_pos[1]:.4f}  zoom={self.home_zoom:.2f}"
            )
        except Exception as e:
            print(f"[PTZ] save_home error: {e}")

    def go_home(self):
        """Non-blocking: slew back to saved home pan/tilt then zoom to saved home zoom.

        Cancels any currently-running go_home by incrementing the generation
        counter — the old thread detects the mismatch and exits early without
        clearing ``_zooming``, so the new thread owns that flag.
        """
        self._homing = True   # set synchronously so _mover sees it before AbsoluteMove fires
        self._go_home_gen += 1
        my_gen = self._go_home_gen
        threading.Thread(target=self._do_go_home, args=(my_gen,),
                         daemon=True, name="ptz-go-home").start()

    def _restore_zoom(self, target_zoom, gen):
        """Drive zoom to target_zoom; up to 3 correction attempts using _poll_pose feedback.

        Exits early if superseded by a newer go_home call (gen mismatch).
        """
        zoom_tol = 0.01
        for attempt in range(3):
            if self._go_home_gen != gen:
                return   # superseded
            dz = target_zoom - self.zoom_pos
            if abs(dz) <= zoom_tol:
                print(f"[PTZ] zoom at {self.zoom_pos:.3f} (target {target_zoom:.3f}) ✓")
                break
            direction = 1.0 if dz > 0 else -1.0
            burst_s = max(0.4, abs(dz) * self.zoom_full_s)
            print(f"[PTZ] zoom restore attempt {attempt + 1}: dz={dz:.3f}  burst={burst_s:.2f}s")
            self._zoom_burst(direction, burst_s)

        else:
            print(f"[PTZ] zoom restore ended at {self.zoom_pos:.3f} (target {target_zoom:.3f})")
        self.zoom_pos = target_zoom   # reconcile internal state to intended target

    def _do_go_home(self, gen):
        """Slew to saved home position and zoom.  Exits early if superseded (gen mismatch)."""
        self._zooming = True
        try:
            if self.ptz:
                try:
                    with self._lock:
                        self.ptz.Stop({'ProfileToken': self.token,
                                       'PanTilt': True, 'Zoom': False})
                except Exception:
                    pass
            if self._go_home_gen != gen:
                return   # superseded before we even started slewing
            if self.ptz and self.home_pos is not None:
                self._return_to(self.home_pos[0], self.home_pos[1], gen)
                if self._go_home_gen != gen:
                    return
                target_zoom = (
                    self._return_zoom if self._return_zoom is not None
                    else self.home_zoom if self.home_zoom is not None
                    else self.zoom_search_pos
                )
                self._restore_zoom(target_zoom, gen)
                if self._go_home_gen != gen:
                    return
                self._return_zoom = None   # consumed; next go_home falls back to home_zoom
            else:
                print("[PTZ] go_home: no home saved; staying at current position/zoom")
        except Exception as e:
            print(f"[PTZ] go_home error: {e}")
        finally:
            # Only clear flags if we are still the active generation.
            # A superseded thread must NOT clear them — the newer thread owns the flags.
            if self._go_home_gen == gen:
                self._zooming = False
                self._homing  = False

    def _absolute_move(self, pan, tilt) -> bool:
        """Issue an AbsoluteMove to (pan, tilt) at full speed.

        Tries with Speed field first (camera may require it), then without.
        Returns True on success, False if both forms fail.
        """
        if not self.ptz:
            return False
        for include_speed in (True, False):
            try:
                req = self.ptz.create_type('AbsoluteMove')
                req.ProfileToken = self.token
                req.Position = {
                    'PanTilt': {'x': float(pan),  'y': float(tilt)},
                    'Zoom':    {'x': float(self.zoom_pos)},
                }
                if include_speed:
                    req.Speed = {'PanTilt': {'x': 1.0, 'y': 1.0}, 'Zoom': {'x': 1.0}}
                with self._lock:
                    self.ptz.AbsoluteMove(req)
                return True
            except Exception as e:
                if not include_speed:
                    print(f"[PTZ] AbsoluteMove failed: {e}")
        return False

    def _return_to(self, pan_target, tilt_target, gen):
        """Slew to (pan_target, tilt_target) and wait for physical arrival.

        Uses AbsoluteMove for a firmware-driven slew at full speed; falls back to
        an iterative RelativeMove loop if AbsoluteMove is not supported.
        Holds ``_zooming=True`` throughout so the detect thread skips frames.
        Exits early if superseded by a newer go_home call (gen mismatch).
        """
        if not self.ptz:
            return
        print(f"[PTZ] Returning home: pan={pan_target:.4f}  tilt={tilt_target:.4f}")
        deadline = time.time() + self.home_timeout

        if self._absolute_move(pan_target, tilt_target):
            while time.time() < deadline:
                if self._go_home_gen != gen:
                    return   # superseded
                try:
                    with self._lock:
                        status = self.ptz.GetStatus({'ProfileToken': self.token})
                    pt = status.Position.PanTilt
                    ep = pan_target  - float(pt.x)
                    et = tilt_target - float(pt.y)
                    if abs(ep) < self.home_tol and abs(et) < self.home_tol:
                        print("[PTZ] Home reached")
                        break
                except Exception as e:
                    print(f"[PTZ] _return_to GetStatus error: {e} — retrying")
                time.sleep(0.1)
            else:
                print("[PTZ] go_home: timed out — stopping wherever camera is")
        else:
            while time.time() < deadline:
                if self._go_home_gen != gen:
                    return   # superseded
                try:
                    with self._lock:
                        status = self.ptz.GetStatus({'ProfileToken': self.token})
                    pt = status.Position.PanTilt
                    ep = pan_target  - float(pt.x)
                    et = tilt_target - float(pt.y)
                except Exception as e:
                    print(f"[PTZ] _return_to GetStatus error: {e} — retrying")
                    time.sleep(0.1)
                    continue
                if abs(ep) < self.home_tol and abs(et) < self.home_tol:
                    print("[PTZ] Home reached")
                    break
                try:
                    req = self.ptz.create_type('RelativeMove')
                    req.ProfileToken = self.token
                    req.Translation  = {
                        'PanTilt': {'x': float(ep), 'y': float(et)},
                        'Zoom':    {'x': 0.0},
                    }
                    req.Speed = {'PanTilt': {'x': 1.0, 'y': 1.0}, 'Zoom': {'x': 1.0}}
                    with self._lock:
                        self.ptz.RelativeMove(req)
                except Exception as e:
                    print(f"[PTZ] _return_to RelativeMove error: {e} — retrying")
                time.sleep(0.4)
            else:
                print("[PTZ] go_home: timed out — stopping wherever camera is")

        try:
            with self._lock:
                self.ptz.Stop({'ProfileToken': self.token, 'PanTilt': True, 'Zoom': False})
        except Exception:
            pass

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def stop_all(self):
        """Stop the mover thread and issue a PTZ Stop command."""
        self._run = False
        if not self.ptz:
            return
        try:
            with self._lock:
                self.ptz.Stop({'ProfileToken': self.token, 'PanTilt': True, 'Zoom': True})
        except Exception:
            pass
