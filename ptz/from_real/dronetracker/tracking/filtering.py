"""Track filtering for the offline and live pipelines.

Single shared design
--------------------
Both pipelines use **`TrackFilter`** — a per-frame streaming filter that judges
each active track on its life-so-far against these criteria:

  1.  Lifespan  - track must have been seen in >= min_lifespan_frames frames.
  2.  Jump      - no displacement > jump_frac * sqrt(W²+H²) within any window
                  of <= jump_window_frames consecutive recorded frames.
  3.  Jitter    - median consecutive-frame step must be <= max_median_step_px.
  3b. Teleport  - no single consecutive-frame step may exceed max_step_px.
  4.  Contrast  - median blob-vs-sky contrast must be >= min_blob_contrast.
  5.  Coast gap - a near-straight track with a >= max_coast_gap_frames missed-
                  detection gap in the recent window is a coasting phantom.

(Criteria 3b/4/5 are opt-in: each is skipped when its config threshold is 0.)

A track that fails is simply not returned from ``update()`` — the caller stops
drawing it.  A track can recover (e.g. once a teleport slides out of the jump
window), but in practice the criteria are designed to be monotonically
satisfied by real drones once they have enough history.

The core math lives in ``_evaluate_trail()`` so offline and online are
guaranteed to use identical logic.

Two-pass helpers (kept for run_seg030_with_mask.py)
---------------------------------------------------
``TrackRecorder``, ``select_surviving_tracks``, and ``draw_filtered_tracks``
remain for use by untracked standalone scripts.  The main pipelines
(``OfflineRunner``, ``LivePtzPipeline``) no longer use them.
"""

__all__ = [
    "TrackFilter",
    # Two-pass helpers (standalone scripts only):
    "TrackRecorder",
    "select_surviving_tracks",
    "draw_filtered_tracks",
]

import math

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Shared criterion evaluator
# ──────────────────────────────────────────────────────────────────────────────

def _trail_straightness(trail):
    """Net displacement / path length of a trail; ~1.0 = near-straight, 0 = static.

    ``trail`` is a list of ``(frame_idx, cx, cy)``.  Returns 0.0 for trails with
    no movement (so a stationary track never reads as "hard linear").
    """
    if len(trail) < 2:
        return 0.0
    path = sum(
        math.hypot(trail[i][1] - trail[i - 1][1], trail[i][2] - trail[i - 1][2])
        for i in range(1, len(trail))
    )
    if path <= 1e-9:
        return 0.0
    net = math.hypot(trail[-1][1] - trail[0][1], trail[-1][2] - trail[0][2])
    return net / path


def _trail_avg_speed(trail):
    """Average speed (px per frame) across a trail, robust to frame gaps.

    Net first->last displacement divided by the frame span.  Used to reject
    near-stationary tracks: a barely-moving blob is not a "hard linear" flight
    even if its net/path ratio is high.
    """
    if len(trail) < 2:
        return 0.0
    span = trail[-1][0] - trail[0][0]
    if span <= 0:
        return 0.0
    net = math.hypot(trail[-1][1] - trail[0][1], trail[-1][2] - trail[0][2])
    return net / span


def _result(kept, n, reason, max_jump=None, median_step=None):
    """Build the ``(kept, info)`` return tuple shared by every evaluation path."""
    return kept, {
        "lifespan":    n,
        "max_jump":    max_jump,
        "median_step": median_step,
        "reason":      reason,
    }


def _evaluate_trail(trail, jump_limit, cfg):
    """Evaluate a centroid trail against the filter criteria.

    Args:
        trail:      List of ``(frame_idx, cx, cy)`` tuples, ordered by
                    frame_idx ascending.
        jump_limit: Pre-computed maximum jump distance in pixels
                    (``cfg.jump_frac * hypot(frame_w, frame_h)``).
        cfg:        A ``TrackFilterConfig`` (or any object with the same
                    attributes).

    Returns:
        ``(kept, info)`` where:
          kept - True if the trail passes all criteria.
          info - dict with keys ``lifespan``, ``max_jump``, ``median_step``,
                 ``reason`` (values are None for criteria not reached).
    """
    n = len(trail)

    # --- Criterion 1: lifespan ---
    if n < cfg.min_lifespan_frames:
        return _result(False, n, f"lifespan {n} < {cfg.min_lifespan_frames}")

    # --- Criterion 2: jump (sliding window) ---
    max_jump = 0.0
    j = 0
    for i in range(n):
        if j < i:
            j = i
        # Rewind j if a previous anchor advanced it past the current window.
        while j > i and (trail[j][0] - trail[i][0]) > cfg.jump_window_frames:
            j -= 1
        while j + 1 < n and (trail[j + 1][0] - trail[i][0]) <= cfg.jump_window_frames:
            j += 1
        if j > i:
            xi, yi = trail[i][1], trail[i][2]
            xj, yj = trail[j][1], trail[j][2]
            dist = math.hypot(xj - xi, yj - yi)
            if dist > max_jump:
                max_jump = dist

    if max_jump > jump_limit:
        return _result(
            False, n,
            f"jump {max_jump:.1f} > {jump_limit:.1f} ({cfg.jump_frac:.2f} x diag)",
            max_jump=max_jump,
        )

    # --- Criterion 3: median consecutive step ---
    steps = [
        math.hypot(trail[i][1] - trail[i - 1][1],
                   trail[i][2] - trail[i - 1][2])
        for i in range(1, n)
    ]
    med_step = float(np.median(steps)) if steps else 0.0

    if med_step > cfg.max_median_step_px:
        return _result(
            False, n,
            f"median_step {med_step:.1f} > {cfg.max_median_step_px:.1f}",
            max_jump=max_jump, median_step=med_step,
        )

    # --- Criterion 3b: max single-frame step (teleport rejection) ---
    max_step_limit = getattr(cfg, "max_step_px", 0.0)
    if max_step_limit and max_step_limit > 0 and steps:
        max_step = max(steps)
        if max_step > max_step_limit:
            return _result(
                False, n,
                f"max_step {max_step:.1f} > {max_step_limit:.1f}",
                max_jump=max_jump, median_step=med_step,
            )

    # --- Criterion 4: blob-vs-sky contrast (drift / sky-track rejection) ---
    min_contrast = getattr(cfg, "min_blob_contrast", 0.0)
    if min_contrast and min_contrast > 0:
        contrasts = [t[3] for t in trail if len(t) > 3 and t[3] is not None]
        need = getattr(cfg, "min_contrast_samples", 8)
        if len(contrasts) >= need:
            med_contrast = float(np.median(contrasts))
            if med_contrast < min_contrast:
                return _result(
                    False, n,
                    f"blob_contrast {med_contrast:.1f} < {min_contrast:.1f}",
                    max_jump=max_jump, median_step=med_step,
                )

    # --- Criterion 5: linear coast gap (recent, windowed) ---
    # A track that is near-straight AND has a >= max_coast_gap_frames consecutive
    # missed-detection gap within the recent window is a coasting phantom (Norfair
    # extrapolated a straight line across frames with no real detection).  Only
    # counts while the gap is recent — once enough clean detections accumulate the
    # gap slides out of the window and the track is allowed back.
    max_coast_gap = int(getattr(cfg, "max_coast_gap_frames", 0))
    if max_coast_gap > 0:
        window  = getattr(cfg, "coast_window_frames", 15)
        last_fidx = trail[-1][0]
        recent  = [p for p in trail if p[0] > last_fidx - window]
        gap = 0
        for i in range(1, len(recent)):
            g = recent[i][0] - recent[i - 1][0] - 1   # missed frames between detections
            if g > gap:
                gap = g
        if gap >= max_coast_gap:
            straightness = _trail_straightness(recent)
            avg_speed    = _trail_avg_speed(recent)
            min_speed    = getattr(cfg, "coast_min_speed_px", 1.0)
            straight_min = getattr(cfg, "coast_linear_straightness", 0.95)
            # Near-zero speed never counts as linear, even at high straightness.
            if avg_speed >= min_speed and straightness >= straight_min:
                return _result(
                    False, n,
                    f"coast_gap {gap} >= {max_coast_gap} & straightness "
                    f"{straightness:.2f} >= {straight_min:.2f} & speed "
                    f"{avg_speed:.1f} >= {min_speed:.1f}",
                    max_jump=max_jump, median_step=med_step,
                )

    return _result(True, n, "ok", max_jump=max_jump, median_step=med_step)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming filter (used by BOTH offline and live pipelines)
# ──────────────────────────────────────────────────────────────────────────────

class TrackFilter:
    """Per-frame streaming track filter — identical in offline and live pipelines.

    Call ``update(tracked_objects)`` once per frame after ``tracker.update()``.
    It returns the set of track IDs that currently pass all criteria
    (judged on their life-so-far).  Pass that set to the draw layer — tracks
    not in the set are simply not drawn.

    Call ``reset()`` whenever the Norfair tracker is recycled so stale
    per-track history doesn't pollute new IDs.

    Memory is bounded: each trail is capped at ``trail_cap`` points, and
    tracks not seen for ``prune_after_frames`` frames are pruned.

    Args:
        frame_w:            Frame width in pixels.
        frame_h:            Frame height in pixels.
        cfg:                A ``TrackFilterConfig``.
        prune_after_frames: Remove a track's history after this many frames
                            without a detection.  Default 300.
    """

    def __init__(self, frame_w, frame_h, cfg, prune_after_frames=300, fps=25.0):
        self._jump_limit     = cfg.jump_frac * math.hypot(frame_w, frame_h)
        self._cfg            = cfg
        self._prune_after    = prune_after_frames
        self._trail_cap      = max(cfg.jump_window_frames * 3, 60)
        self._frame_idx      = 0
        self._records        = {}   # {track_id: [(frame_idx, cx, cy, contrast), ...]}
        self._last_seen      = {}   # {track_id: last frame_idx with detection}
        # Stationary-kill: drop a track that stays within stationary_move_px for
        # stationary_kill_s seconds.  _stationary[id] = [ref_x, ref_y, frames].
        self._stationary       = {}
        self._stationary_move  = float(getattr(cfg, "stationary_move_px", 0.0))
        self._max_stationary   = int(getattr(cfg, "stationary_kill_s", 0.0) * fps)

    def update(self, tracked_objects):
        """Evaluate all currently active tracks and return the passing set.

        Args:
            tracked_objects: Iterable of Norfair TrackedObject instances from
                             ``tracker.update()``.

        Returns:
            set of track IDs that currently pass the filter.
            When ``cfg.enabled`` is False, returns every active confirmed ID
            (i.e. the filter is a transparent no-op).
        """
        self._frame_idx += 1
        fidx = self._frame_idx

        # Gather IDs with an active detection this frame.
        active_ids = set()
        for obj in tracked_objects:
            if obj.last_detection is None:
                continue
            tid = obj.id
            active_ids.add(tid)
            cx = float(obj.estimate[0][0])
            cy = float(obj.estimate[0][1])
            con = getattr(obj.last_detection, "contrast", None)   # real-det blob contrast
            trail = self._records.setdefault(tid, [])
            trail.append((fidx, cx, cy, con))
            if len(trail) > self._trail_cap:
                trail.pop(0)
            self._last_seen[tid] = fidx

            # Stationary tracking: count consecutive frames within move radius of
            # a fixed reference; reset the reference when the track leaves it.
            if self._max_stationary > 0:
                st = self._stationary.get(tid)
                if st is None:
                    self._stationary[tid] = [cx, cy, 0]
                elif math.hypot(cx - st[0], cy - st[1]) <= self._stationary_move:
                    st[2] += 1
                else:
                    st[0], st[1], st[2] = cx, cy, 0

        # Prune tracks that have been gone too long.
        stale = [
            tid for tid, last in self._last_seen.items()
            if fidx - last > self._prune_after
        ]
        for tid in stale:
            self._records.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._stationary.pop(tid, None)

        # Short-circuit: when disabled pass everything.
        if not self._cfg.enabled:
            return active_ids

        # Evaluate each active track on its life-so-far.
        survivors = set()
        for tid in active_ids:
            trail = self._records.get(tid, [])
            kept, _ = _evaluate_trail(trail, self._jump_limit, self._cfg)
            # Stationary-kill: drop a track that hasn't moved for too long.
            if kept and self._max_stationary > 0:
                st = self._stationary.get(tid)
                if st is not None and st[2] >= self._max_stationary:
                    kept = False
            if kept:
                survivors.add(tid)

        return survivors

    def reset(self):
        """Clear all per-track history (call when the Norfair tracker is recycled)."""
        self._frame_idx = 0
        self._records.clear()
        self._last_seen.clear()
        self._stationary.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Two-pass helpers (kept for run_seg030_with_mask.py and similar scripts)
# ──────────────────────────────────────────────────────────────────────────────

class TrackRecorder:
    """Collect per-track centroid trails during a single offline pass.

    Call ``observe(frame_idx, tracked_objects)`` once per frame after
    ``tracker.update()``.  ``records`` is built incrementally and can be
    passed directly to ``select_surviving_tracks`` after the pass completes.

    Attributes:
        records: dict mapping track_id -> list of (frame_idx, cx, cy) tuples.
    """

    def __init__(self):
        self.records = {}   # {track_id: [(frame_idx, cx, cy), ...]}

    def observe(self, frame_idx, tracked_objects):
        """Record the current position of every confirmed track.

        Args:
            frame_idx:       Zero-based frame index (int).
            tracked_objects: Iterable of Norfair TrackedObject instances from
                             ``tracker.update()``.
        """
        for obj in tracked_objects:
            if obj.last_detection is None:
                continue
            cx = float(obj.estimate[0][0])
            cy = float(obj.estimate[0][1])
            self.records.setdefault(obj.id, []).append((frame_idx, cx, cy))


def select_surviving_tracks(records, frame_w, frame_h, cfg):
    """Apply the three whole-life filter criteria and return surviving IDs.

    Args:
        records:   dict {track_id: [(frame_idx, cx, cy), ...]} from TrackRecorder.
        frame_w:   Frame width in pixels.
        frame_h:   Frame height in pixels.
        cfg:       A TrackFilterConfig.

    Returns:
        (survivors, diagnostics) where:
          survivors   - set of track IDs that pass all criteria.
          diagnostics - dict {track_id: {lifespan, max_jump, median_step,
                                         kept, reason}}.
    """
    diagonal   = math.hypot(frame_w, frame_h)
    jump_limit = cfg.jump_frac * diagonal

    survivors   = set()
    diagnostics = {}

    for tid, trail in records.items():
        kept, info = _evaluate_trail(trail, jump_limit, cfg)
        diagnostics[tid] = {**info, "kept": kept}
        if kept:
            survivors.add(tid)

    return survivors, diagnostics


def draw_filtered_tracks(frame, frame_idx, records, survivors):
    """Draw surviving tracks onto ``frame`` for a two-pass offline script.

    Mirrors the look of ``_default_draw`` (green dot + ID label + blue trail)
    but reconstructs each trail from the pass-1 ``records`` up to
    ``frame_idx``.  Only draws a track on frames where Norfair reported an
    active detection (exact-frame match), mirroring ``_default_draw``'s
    behaviour.

    Does NOT touch the ``track_history`` singleton.

    Args:
        frame:      BGR image to draw onto (modified in-place).
        frame_idx:  Zero-based index of the current frame.
        records:    dict {track_id: [(frame_idx, cx, cy), ...]} from TrackRecorder.
        survivors:  set of track IDs to draw (from select_surviving_tracks).

    Returns:
        The modified ``frame``.
    """
    for tid in survivors:
        trail_data = records.get(tid)
        if not trail_data:
            continue

        trail_pts = [(int(round(cx)), int(round(cy)))
                     for fidx, cx, cy in trail_data
                     if fidx <= frame_idx]

        if not trail_pts:
            continue

        # Only draw if the most-recent recorded detection is at this frame.
        last_recorded_fidx = trail_data[len(trail_pts) - 1][0]
        if last_recorded_fidx != frame_idx:
            continue

        x, y = trail_pts[-1]
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {tid}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(trail_pts) > 1:
            cv2.polylines(frame, [np.array(trail_pts, dtype=np.int32)],
                          False, (255, 0, 0), 2)

    return frame
