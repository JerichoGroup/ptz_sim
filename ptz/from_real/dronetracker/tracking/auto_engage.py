# -*- coding: utf-8 -*-
"""Automatic IDLE -> TRACK ("auto-hunt") gate from motion history.

At long range appearance is useless (a target is a few pixels); *how a track
moves* is what separates a drone from a bird.  Birds fly irregularly — flapping
gives fluctuating speed, frequent heading changes, high curvature, and an
oscillating apparent size — while drones are smooth and rigid.  This module turns
that into a single ``drone_score`` in ``[0, 1]`` (1 = very drone-like) computed
from each track's recent centroid trail and apparent-area sequence.

Grounding: Nesteruk et al. (*Sensors* 2026) and Yao et al. (*Drones* 2026) both
show motion trajectory is the discriminator; Yao et al. reach 93%+ per modality
on real single-camera footage with exactly these handcrafted kinematic + area
fluctuation features.

Design notes
------------
* **Pure-ish & testable.** The feature maths are module-level functions over
  plain ``(t, cx, cy, area)`` tuples — no cv2 / norfair / torch imports — so they
  unit-test without a camera.  ``score_drone_likeness`` is the single pluggable
  seam: swap in a trained RF/SVM/BiLSTM later without touching the plumbing.
* **Scale-normalized.** Positional features are divided by an apparent-size
  diagonal so the gate is zoom-invariant (the camera zooms on lock-on).
* **Coasting excluded.** Samples are appended only on real detections — holding a
  position constant during a detection miss is the worst corruptor of kinematic
  features (Yao et al. §5.5).
* **Detect-thread owned.** One instance lives on the detect worker; no
  cross-thread state.  ``update`` stamps ``obj.drone_score`` on each scored track
  so the (shallow-copied) published tracks carry the score to the renderer.
"""

from __future__ import annotations

__all__ = [
    "AutoEngageEvaluator",
    "FeatureLogger",
    "compute_features",
    "score_features",
    "FEATURE_NAMES",
]

import atexit
import csv
import math
from collections import deque
from typing import Optional

import numpy as np

# Order of features in the logged CSV and the score.  Keep stable.
FEATURE_NAMES = [
    "velocity_cv",
    "accel_mean",
    "heading_change_ratio",
    "curvature_cv",
    "curvature_mean",
    "area_cv",
    "area_change_mean",
    "area_change_max",
    "net_disp_frac",
    "katz_fd",
]

# Below this many real samples a track has too little history to characterize.
_MIN_SAMPLES = 5
# Heading is considered "changed" between two steps when it turns more than this.
_HEADING_EPS_RAD = math.radians(8.0)
# A track that has travelled less than this (× its apparent-size diagonal) over
# the window is treated as static: its fluctuation features are all ~0, which
# would otherwise read as "perfectly rigid = ideal drone".  A drone in flight
# moves; a static hot pixel / stuck blob is not a target.  Independent of the
# (tunable) tier-1 ``min_net_disp_frac`` engage gate.
_MIN_MOTION_FRAC = 0.1


# ---------------------------------------------------------------------------
# Pure feature maths (no cv2 / norfair) — unit-testable in isolation
# ---------------------------------------------------------------------------

def _cv(values) -> float:
    """Coefficient of variation (std / mean); 0 when mean is ~0 or n < 2."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    mean = float(arr.mean())
    if abs(mean) < 1e-9:
        return 0.0
    return float(arr.std() / mean)


def _size_diag(areas) -> float:
    """Apparent-size diagonal used to scale-normalize positional features.

    Approximates the bbox diagonal from the contour area assuming a roughly
    square blob (diag ≈ sqrt(2 * area)).  Clamped to >= 1 px so we never divide
    by zero for sub-pixel blobs.
    """
    a = [x for x in areas if x is not None and x > 0]
    if not a:
        return 1.0
    return max(1.0, math.sqrt(2.0 * float(np.mean(a))))


def _step_speeds(pts, diag: float):
    """Per-step displacement magnitudes, normalized by ``diag``."""
    speeds = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        speeds.append(math.hypot(x1 - x0, y1 - y0) / diag)
    return speeds


def _headings(pts):
    """Heading angle (radians) of each step vector; skips zero-length steps."""
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        if dx == 0.0 and dy == 0.0:
            continue
        out.append(math.atan2(dy, dx))
    return out


def _heading_change_ratio(headings) -> float:
    """Fraction of consecutive heading pairs that turn more than the epsilon."""
    if len(headings) < 2:
        return 0.0
    changes = 0
    for a, b in zip(headings[:-1], headings[1:]):
        d = abs(math.atan2(math.sin(b - a), math.cos(b - a)))  # wrapped diff
        if d > _HEADING_EPS_RAD:
            changes += 1
    return changes / (len(headings) - 1)


def _curvatures(pts):
    """Turn angle (radians, 0..pi) between successive step vectors."""
    out = []
    for i in range(len(pts) - 2):
        ax, ay = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        bx, by = pts[i + 2][0] - pts[i + 1][0], pts[i + 2][1] - pts[i + 1][1]
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = (ax * bx + ay * by) / (na * nb)
        out.append(math.acos(max(-1.0, min(1.0, cosang))))
    return out


def _area_change(areas):
    """(mean, max) of |A_i - A_{i-1}| / A_i over the area sequence."""
    a = [x for x in areas if x is not None and x > 0]
    if len(a) < 2:
        return 0.0, 0.0
    rels = [abs(a[i] - a[i - 1]) / a[i] for i in range(1, len(a))]
    return float(np.mean(rels)), float(np.max(rels))


def _net_disp_frac(pts, diag: float) -> float:
    """Max distance of any point from the first point, normalized by ``diag``."""
    if len(pts) < 2:
        return 0.0
    x0, y0 = pts[0]
    return max(math.hypot(x - x0, y - y0) for x, y in pts) / diag


def _katz_fd(pts, diag: float) -> float:
    """Katz fractal dimension (trajectory tortuosity); 1 = straight line.

    D = log10(N) / (log10(N) + log10(d / L)) where L is total path length and d
    is the max distance from the first point.  Birds (meandering) score higher.
    """
    if len(pts) < 3:
        return 0.0
    length = sum(math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                 for p0, p1 in zip(pts[:-1], pts[1:])) / diag
    d = _net_disp_frac(pts, diag)
    if length <= 0 or d <= 0:
        return 0.0
    n = len(pts)
    denom = math.log10(n) + math.log10(d / length)
    if abs(denom) < 1e-9:
        return 0.0
    return math.log10(n) / denom


def compute_features(samples) -> Optional[dict]:
    """Compute the drone-likeness feature dict from a track's history samples.

    Args:
        samples: iterable of ``(t, cx, cy, area)`` tuples (``area`` may be None).

    Returns:
        Dict keyed by ``FEATURE_NAMES``, or ``None`` if there is too little
        history to characterize the motion (< ``_MIN_SAMPLES`` points).
    """
    samples = list(samples)
    if len(samples) < _MIN_SAMPLES:
        return None

    pts   = [(float(s[1]), float(s[2])) for s in samples]
    areas = [s[3] for s in samples]
    diag  = _size_diag(areas)

    speeds     = _step_speeds(pts, diag)
    headings   = _headings(pts)
    curvatures = _curvatures(pts)
    accel      = [abs(speeds[i] - speeds[i - 1]) for i in range(1, len(speeds))]
    a_change_mean, a_change_max = _area_change(areas)

    return {
        "velocity_cv":          _cv(speeds),
        "accel_mean":           float(np.mean(accel)) if accel else 0.0,
        "heading_change_ratio": _heading_change_ratio(headings),
        "curvature_cv":         _cv(curvatures),
        "curvature_mean":       float(np.mean(curvatures)) if curvatures else 0.0,
        "area_cv":              _cv([a for a in areas if a is not None and a > 0]),
        "area_change_mean":     a_change_mean,
        "area_change_max":      a_change_max,
        "net_disp_frac":        _net_disp_frac(pts, diag),
        "katz_fd":              _katz_fd(pts, diag),
    }


def score_features(feats: dict, cfg) -> float:
    """Combine fluctuation features into a drone-likeness score in [0, 1].

    Each fluctuation feature ``f`` with ceiling ``c`` contributes
    ``clip(1 - f/c, 0, 1)`` — 1 when the motion is perfectly rigid (f=0,
    drone-like), 0 once it reaches the bird-like ceiling.  The score is the mean
    of those contributions.  Ceilings of 0 disable that feature.

    A track that has barely moved abstains (score 0): with no real motion the
    fluctuation features are all ~0 and would otherwise read as a perfect drone.
    """
    if feats.get("net_disp_frac", 0.0) < _MIN_MOTION_FRAC:
        return 0.0
    pairs = (
        (feats["velocity_cv"],          cfg.vel_cv_max),
        (feats["accel_mean"],           cfg.accel_mean_max),
        (feats["heading_change_ratio"], cfg.heading_change_max),
        (feats["curvature_cv"],         cfg.curvature_cv_max),
        (feats["curvature_mean"],       cfg.curvature_mean_max),
        (feats["area_cv"],              cfg.area_cv_max),
        (feats["area_change_mean"],     cfg.area_change_max),
    )
    contribs = [max(0.0, min(1.0, 1.0 - f / c)) for f, c in pairs if c > 0]
    if not contribs:
        return 0.0
    return sum(contribs) / len(contribs)


# ---------------------------------------------------------------------------
# Helpers bridging norfair / ground-mask objects to the pure maths
# ---------------------------------------------------------------------------

def detection_area(det) -> Optional[float]:
    """Apparent area of a norfair Detection: ``.area`` if stashed, else from box."""
    a = getattr(det, "area", None)
    if a is not None:
        return float(a)
    box = getattr(det, "box", None)
    if box is not None and len(box) >= 4:
        x1, y1, x2, y2 = box[:4]
        return float(abs((x2 - x1) * (y2 - y1)))
    return None


def point_in_sky(gmg, x: float, y: float) -> bool:
    """True if (x, y) is in the sky region (ground mask == 0) or no mask exists.

    The ground mask is 255 over ground, 0 over sky.  Missing mask / out-of-bounds
    is treated as sky so we never wrongly suppress a candidate.
    """
    if gmg is None:
        return True
    mask = getattr(gmg, "mask", None)
    if mask is None:
        return True
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return True
    return bool(mask[yi, xi] == 0)


# ---------------------------------------------------------------------------
# Feature logger (builds the dataset to later train a learned scorer)
# ---------------------------------------------------------------------------

class FeatureLogger:
    """Append per-track feature rows to a CSV (header written once)."""

    def __init__(self, path: str):
        self._path = path
        self._fh = None
        self._writer = None

    def _ensure(self) -> None:
        if self._writer is not None:
            return
        import os
        new = (not os.path.exists(self._path)) or os.path.getsize(self._path) == 0
        self._fh = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if new:
            self._writer.writerow(["t", "track_id"] + FEATURE_NAMES + ["score", "outcome"])
        # The live detect thread is a daemon with no shutdown hook; ensure the
        # handle is flushed/closed at process exit (offline calls close() itself).
        atexit.register(self.close)

    def log(self, t: float, track_id, feats: dict, score: float, outcome: str) -> None:
        self._ensure()
        row = [f"{t:.3f}", track_id]
        row += [f"{feats[name]:.5f}" for name in FEATURE_NAMES]
        row += [f"{score:.5f}", outcome]
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class AutoEngageEvaluator:
    """Score tracks and pick the best drone-like candidate to auto-engage.

    Owned by a single thread.  Call :meth:`update` once per frame with the
    (post-TrackFilter) tracks; it stamps ``obj.drone_score`` on each scored
    track and returns the id of the highest-scoring track that has passed the
    structural pre-gates and the score threshold continuously for ``dwell_s``
    (or ``None``).  The caller decides whether to act on the returned id (live
    auto-engage is additionally gated by ``cfg.enabled`` and a cooldown).
    """

    def __init__(self, cfg, logger: Optional[FeatureLogger] = None):
        self._cfg = cfg
        self._logger = logger
        self._history: dict = {}           # track id -> deque[(t, cx, cy, area)]
        self._first_qualified: dict = {}   # track id -> time it first qualified
        self._engaged_id = None            # latched would-zoom target (sticky lock)
        self._engaged_missing = 0          # consecutive frames the target was undetected

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()

    def reset(self) -> None:
        """Drop all per-track history and dwell timers.

        Called on a TRACK->IDLE reset (the Norfair tracker is rebuilt there, so
        IDs restart): every re-acquired target must re-prove itself from scratch
        rather than inherit a satisfied dwell from before the lock.
        """
        self._history.clear()
        self._first_qualified.clear()
        self._engaged_id = None
        self._engaged_missing = 0

    # -- internal ----------------------------------------------------------

    def _prune_old(self, hist: deque, now: float) -> None:
        horizon = now - self._cfg.history_s
        while hist and hist[0][0] < horizon:
            hist.popleft()

    def _tier1(self, hist, cx: float, cy: float, gmg) -> tuple:
        """Cheap structural pre-gates.  Returns (passed, reason)."""
        cfg = self._cfg
        if len(hist) < cfg.min_detections:
            return False, "few_samples"
        if cfg.require_sky and not point_in_sky(gmg, cx, cy):
            return False, "not_sky"
        pts   = [(h[1], h[2]) for h in hist]
        areas = [h[3] for h in hist]
        if _net_disp_frac(pts, _size_diag(areas)) < cfg.min_net_disp_frac:
            return False, "static"
        return True, "ok"

    # -- public ------------------------------------------------------------

    def update(self, tracks, gmg, now: float) -> Optional[int]:
        """Ingest one frame of tracks; return the best engageable id or None."""
        cfg = self._cfg
        present = set()
        detected_objs = {}   # id -> obj (has a real detection this frame)
        candidates = []      # (score, id, obj)

        for obj in tracks:
            tid = obj.id
            if tid is None:
                continue
            present.add(tid)                 # keep history alive across brief misses
            if obj.last_detection is None:   # coasting — present but add no sample
                obj.auto_engage = False      # never leave a stale would-zoom flag
                continue
            detected_objs[tid] = obj

            cx = float(obj.estimate[0][0])
            cy = float(obj.estimate[0][1])
            area = detection_area(obj.last_detection)

            hist = self._history.get(tid)
            if hist is None:
                hist = deque(maxlen=cfg.history_max)
                self._history[tid] = hist
            hist.append((now, cx, cy, area))
            self._prune_old(hist, now)

            feats = compute_features(hist)
            if feats is None:
                obj.drone_score = 0.0
                obj.auto_engage = False
                self._first_qualified.pop(tid, None)
                continue

            score = score_features(feats, cfg)
            obj.drone_score = score
            obj.auto_engage = False   # set True below only for the chosen candidate

            passed, reason = self._tier1(hist, cx, cy, gmg)
            engage_ok = passed and score >= cfg.dronelike_threshold

            if engage_ok:
                self._first_qualified.setdefault(tid, now)
                dwell_met = (now - self._first_qualified[tid]) >= cfg.dwell_s
                outcome = "qualified" if dwell_met else "dwelling"
            else:
                self._first_qualified.pop(tid, None)
                dwell_met = False
                outcome = reason if not passed else "low_score"

            if self._logger is not None:
                self._logger.log(now, tid, feats, score, outcome)

            if engage_ok and dwell_met:
                candidates.append((score, tid, obj))

        # Drop bookkeeping for tracks that vanished this frame.
        for tid in [t for t in self._history if t not in present]:
            self._history.pop(tid, None)
            self._first_qualified.pop(tid, None)

        # --- Engagement latch (sticky lock) ---
        # Once a track is chosen, KEEP it as the would-zoom target through transient
        # score dips (e.g. while the drone banks/turns) — mirroring the live
        # pipeline, which stays in TRACK following the target rather than re-judging
        # smoothness every frame.  Release only when the target is lost (undetected
        # for release_frames), so it can then re-acquire a new candidate.
        if self._engaged_id is not None:
            obj = detected_objs.get(self._engaged_id)
            if obj is not None:
                self._engaged_missing = 0
                obj.auto_engage = True
                return self._engaged_id
            # Target not detected this frame (coasting or gone).
            self._engaged_missing += 1
            if self._engaged_missing >= cfg.release_frames:
                self._engaged_id = None
                self._engaged_missing = 0
            else:
                # Still latched, but coasting: its estimate is a Kalman
                # extrapolation that has drifted off any real blob.  Do NOT hand
                # that ghost id to the live lock-on (it would zoom to empty space);
                # return None so the caller waits until the target is re-detected.
                return None

        # Not engaged: acquire a new target via the strict gate (dwell + score).
        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            _best_score, best_id, best_obj = candidates[0]
            self._engaged_id = best_id
            self._engaged_missing = 0
            best_obj.auto_engage = True   # the track the live pipeline would zoom on
            return best_id
        return None
