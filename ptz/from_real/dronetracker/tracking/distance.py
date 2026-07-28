"""Distance functions for Norfair multi-object tracking.

Provides factory functions that return Norfair-compatible callables for the
``distance_function`` and ``reid_distance_function`` Tracker arguments.

The default Norfair ``"euclidean"`` function considers only the predicted
centroid position.  During crossings, two objects that trade positions leave
both their detections within the gate of both predicted tracks, so the greedy
assignment can produce an ID swap.

``make_motion_size_distance`` adds a *capped* size penalty that helps separate
two candidates when their predicted positions converge.  The cap ensures the
size term can only re-order tied candidates — it cannot alone push a
geometrically close match out of the gate.

``make_reid_distance`` supplies a matching signal for Norfair's built-in ReID
mechanism so that a track that briefly loses its detection can reclaim its old
ID rather than spawning a new one.

Both factories optionally take an ``appearance_store`` (see
``dronetracker.tracking.appearance.AppearanceStore``).  When supplied, a
detection whose appearance descriptor is far from the track's stored template
adds a *bounded, dead-banded* penalty — a tie-breaker, like the size term.  A
small ``appearance_deadband`` absorbs an object's normal frame-to-frame
self-variation (so a true match is never penalised) and ``appearance_penalty_cap``
bounds the penalty (so a transient descriptor spike can never reject the
object's own detection — appearance only *re-orders* competing candidates).
A clearly-different object (e.g. a bee crossing a vibrating drone) is thus
out-competed for the track's ID rather than rejected outright.

Both factories are **pure numpy** — no cv2 or norfair imports at module level —
so they are importable in tests without a heavy environment.
"""

__all__ = ["make_motion_size_distance", "make_reid_distance"]

import numpy as np

from dronetracker.tracking.appearance import descriptor_distance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Euclidean distance between two (1, 2) centroid arrays."""
    return float(np.hypot(pts_a[0, 0] - pts_b[0, 0], pts_a[0, 1] - pts_b[0, 1]))


def _size_penalty(
    area_a: float,
    area_b: float,
    size_weight: float,
    size_penalty_cap: float,
) -> float:
    """Penalty based on the area ratio of two detections.

    Returns 0 when the areas are equal; grows with their ratio up to
    ``size_penalty_cap``.
    """
    denom = max(min(area_a, area_b), 1.0)
    ratio = max(area_a, area_b) / denom
    return min((ratio - 1.0) * size_weight, size_penalty_cap)


def _track_descriptor(obj, appearance_store):
    """Best appearance descriptor for a TrackedObject.

    Prefers the stable EMA template from ``appearance_store`` (keyed by track
    ID), falling back to the object's most recent detection embedding.  The
    fallback matters on the ReID path: Norfair passes a not-yet-confirmed
    candidate whose ``id`` is ``None`` (so it has no template yet), but it does
    carry a ``last_detection.embedding``.
    """
    if appearance_store is not None and getattr(obj, "id", None) is not None:
        template = appearance_store.get(obj.id)
        if template is not None:
            return template
    last_det = getattr(obj, "last_detection", None)
    return getattr(last_det, "embedding", None) if last_det is not None else None


def _appearance_penalty(
    emb_a,
    emb_b,
    appearance_weight: float,
    appearance_deadband: float = 0.0,
    appearance_penalty_cap: float = float("inf"),
) -> float:
    """Bounded, dead-banded penalty from two appearance descriptors.

    Returns 0 when either descriptor is missing (no appearance signal) so the
    caller falls back cleanly to position + size.  Otherwise:

        pen = min(appearance_weight * max(0, dist - deadband), cap)

    where ``dist`` is the Bhattacharyya distance (∈ [0, 1]).

    The **deadband** absorbs the normal frame-to-frame self-variation of an
    object's own descriptor (measured 75th-percentile ≈ 0.11), so a true match
    incurs *zero* appearance penalty and is never destabilised.  The **cap**
    bounds the penalty so a transient descriptor spike can never push an
    otherwise-good positional match past the gate — appearance can only
    *re-order* competing candidates, not reject the object's own detection.
    Together these stop appearance-driven track fragmentation.
    """
    if appearance_weight <= 0.0:
        return 0.0
    d = descriptor_distance(emb_a, emb_b)
    if d is None:
        return 0.0
    pen = appearance_weight * max(0.0, d - appearance_deadband)
    return min(pen, appearance_penalty_cap)


def _is_slow(tracked_object, slow_speed: float) -> bool:
    """True if the track's Kalman speed is below ``slow_speed`` (px/frame).

    A slow track barely moves, so position can't distinguish its own detection
    from a nearby intruder — appearance should decide.  Returns False when
    ``slow_speed <= 0`` (disabled) or the object exposes no velocity estimate.
    """
    if slow_speed <= 0.0:
        return False
    vel = getattr(tracked_object, "estimate_velocity", None)
    if vel is None:
        return False
    v = np.asarray(vel, dtype=float).ravel()
    if v.size < 2:
        return False
    return float(np.hypot(v[0], v[1])) < slow_speed


def _effective_cap(
    appearance_store,
    track_id,
    base_cap: float,
    dormant_cap: float,
    maturity_frames: int,
    dormancy_frames: int,
    is_slow: bool,
) -> float:
    """Pick the appearance cap for one track based on its motion state.

    A *mature* track (matched on many frames) gets the strong ``dormant_cap``
    when it is **slow** (low Kalman speed) or **dormant** (no match for
    ``dormancy_frames``).  In both states the object barely moves, so position
    can no longer tell its own detection from a nearby intruder/clutter —
    appearance must decide.  The strong cap lets an appearance-mismatched
    candidate be rejected outright (it can't steal a slow/stopped object's ID),
    while the object's own detection (appearance ≈ template) stays well within
    the gate.  Fast-moving or young tracks keep the bounded ``base_cap``
    (position-dominant, robust to appearance noise / fragmentation).
    """
    if dormant_cap <= base_cap or not hasattr(appearance_store, "is_mature"):
        return base_cap
    if not appearance_store.is_mature(track_id, maturity_frames):
        return base_cap
    if is_slow or appearance_store.frames_since_match(track_id) >= dormancy_frames:
        return dormant_cap
    return base_cap


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def make_motion_size_distance(
    size_weight: float = 40.0,
    size_penalty_cap: float = 60.0,
    appearance_store=None,
    appearance_weight: float = 0.0,
    appearance_deadband: float = 0.0,
    appearance_penalty_cap: float = float("inf"),
    appearance_maturity_frames: int = 45,
    appearance_dormancy_frames: int = 20,
    appearance_dormant_cap: float = float("inf"),
    appearance_slow_speed: float = 0.0,
):
    """Return a Norfair ``distance_function`` that uses position + size [+ appearance].

    The returned callable has the signature ::

        distance(detection, tracked_object) -> float

    as expected by ``norfair.Tracker(distance_function=...)``.  A match is
    accepted when the returned value is strictly below
    ``Tracker.distance_threshold``.

    The primary discrimination signal is the Kalman-predicted centroid distance
    (``tracked_object.estimate``).  The Kalman velocity component means a track
    moving in one direction already prefers detections that continue along that
    direction — this is the main defence against ID-swaps at crossings.

    The size penalty breaks ties when two predicted positions nearly coincide but
    the objects differ in area.  The cap prevents a large area mismatch from
    alone rejecting an otherwise close positional match.

    When ``appearance_store`` is supplied and ``appearance_weight > 0``, a
    bounded appearance penalty is added: the Bhattacharyya distance between the
    detection's descriptor (``detection.embedding``) and the track's stored
    template, dead-banded and capped (see ``_appearance_penalty``).  This lets a
    clearly-different object be out-competed for the track without ever rejecting
    the object's own detection.  Missing descriptor/template ⇒ no appearance term
    (graceful fall-back to position + size).

    Args:
        size_weight:       Multiplier applied to (area_ratio - 1).
                           ``40.0`` adds at most ``size_penalty_cap`` px for a
                           2.5× area mismatch.
        size_penalty_cap:  Hard upper bound on the size penalty (px).
        appearance_store:  Optional ``AppearanceStore`` keyed by track ID.
        appearance_weight: Multiplier (px) on the [0, 1] appearance distance.
                           ``0.0`` (default) disables appearance entirely.
        appearance_slow_speed: Kalman speed (px/frame) below which a mature
                           track is "slow" and appearance switches to the strong
                           gate — its next location is decided by appearance, not
                           by proximity to the prediction / nearby motion.
                           ``0.0`` disables the slow trigger.
    """
    def distance(detection, tracked_object) -> float:
        pos = _pos_distance(detection.points, tracked_object.estimate)
        total = pos

        det_area = getattr(detection, "area", None)
        last_det = tracked_object.last_detection
        trk_area = getattr(last_det, "area", None) if last_det is not None else None

        if det_area is not None and trk_area is not None and det_area > 0 and trk_area > 0:
            total += _size_penalty(det_area, trk_area, size_weight, size_penalty_cap)

        if appearance_store is not None and appearance_weight > 0.0:
            tid      = tracked_object.id
            template = appearance_store.get(tid)
            det_emb  = getattr(detection, "embedding", None)
            is_slow  = _is_slow(tracked_object, appearance_slow_speed)
            eff_cap  = _effective_cap(
                appearance_store, tid, appearance_penalty_cap,
                appearance_dormant_cap, appearance_maturity_frames,
                appearance_dormancy_frames, is_slow,
            )
            total += _appearance_penalty(
                det_emb, template, appearance_weight,
                appearance_deadband, eff_cap,
            )

        return total

    return distance


def make_reid_distance(
    size_weight: float = 40.0,
    size_penalty_cap: float = 60.0,
    appearance_store=None,
    appearance_weight: float = 0.0,
    appearance_deadband: float = 0.0,
    appearance_penalty_cap: float = float("inf"),
    appearance_maturity_frames: int = 45,
    appearance_dormant_cap: float = float("inf"),
):
    """Return a Norfair ``reid_distance_function`` that uses position + size [+ appearance].

    The returned callable has the signature ::

        reid_distance(matched_object, unmatched_object) -> float

    as expected by ``norfair.Tracker(reid_distance_function=...)``.  Both
    arguments are ``TrackedObject`` instances: *matched_object* is a live
    (recently-updated) track; *unmatched_object* is a dead track being
    considered for re-identification.  A match is accepted when the returned
    value is strictly below ``Tracker.reid_distance_threshold``.

    Re-identification lets a track that was briefly starved of detections (e.g.
    during the exact overlap frame of a crossing) reclaim its old ID rather than
    spawning a fresh one.

    When ``appearance_store`` is supplied and ``appearance_weight > 0``, a
    bounded appearance penalty (dead-banded Bhattacharyya distance between the
    two tracks' descriptors) is added, so a dead track is only re-identified by
    an object that also *looks* like it.  For a *mature* dead track the cap is
    raised to ``appearance_dormant_cap`` (strong gate), so a clearly-different
    object cannot reclaim its ID.

    Args:
        size_weight:       Same semantics as ``make_motion_size_distance``.
        size_penalty_cap:  Same semantics as ``make_motion_size_distance``.
        appearance_store:  Optional ``AppearanceStore`` keyed by track ID.
        appearance_weight: Multiplier (px) on the [0, 1] appearance distance.
                           ``0.0`` (default) disables appearance entirely.
    """
    def reid_distance(matched_object, unmatched_object) -> float:
        pos = _pos_distance(matched_object.estimate, unmatched_object.estimate)
        total = pos

        last_matched   = matched_object.last_detection
        last_unmatched = unmatched_object.last_detection
        area_m = getattr(last_matched,   "area", None) if last_matched   is not None else None
        area_u = getattr(last_unmatched, "area", None) if last_unmatched is not None else None

        if area_m is not None and area_u is not None and area_m > 0 and area_u > 0:
            total += _size_penalty(area_m, area_u, size_weight, size_penalty_cap)

        if appearance_store is not None and appearance_weight > 0.0:
            # The "matched" candidate is typically a not-yet-confirmed track
            # (id is None) so it has no template — fall back to its last
            # detection embedding.  The "unmatched" dead track keeps its EMA
            # template until pruned.  A ReID candidate is dormant by definition,
            # so use the dead object's maturity to pick a strong gate (an
            # appearance-mismatched object can't reclaim a mature track's ID).
            desc_m = _track_descriptor(matched_object, appearance_store)
            desc_u = _track_descriptor(unmatched_object, appearance_store)
            eff_cap = _effective_cap(
                appearance_store, getattr(unmatched_object, "id", None),
                appearance_penalty_cap, appearance_dormant_cap,
                appearance_maturity_frames, dormancy_frames=0, is_slow=True,
            )
            total += _appearance_penalty(
                desc_m, desc_u, appearance_weight,
                appearance_deadband, eff_cap,
            )

        return total

    return reid_distance
