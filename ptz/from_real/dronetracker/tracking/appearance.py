"""Visual-appearance descriptors and per-track template store.

Adds an appearance signal to data association so that a clearly-different
object (e.g. a bee crossing a vibrating drone) cannot inherit a track's ID by
position alone.  The flow is:

1. ``compute_descriptor(frame, box)`` turns the image patch under a detection's
   bounding box into a cheap, normalised colour/intensity histogram.  The patch
   descriptor is stashed on the Norfair ``Detection`` as ``det.embedding``
   (see ``dronetracker.detection.clustering.motion_to_detections``).
2. ``AppearanceStore`` keeps one *template* per live track — an exponential
   moving average of that track's matched descriptors.  The template is seeded
   when a track first appears and updated slowly thereafter, so a single
   bad/ambiguous frame cannot poison it.
3. ``dronetracker.tracking.distance`` consults the store: a detection whose
   descriptor is far (Bhattacharyya distance) from the track's template gets a
   *bounded, dead-banded* penalty — a tie-breaker that out-competes a clearly-
   different object for the track without ever rejecting the object's own
   detection (which would fragment the track).

The histogram + distance helpers are **pure numpy** (cv2 is imported locally,
only inside ``compute_descriptor``) so the math is unit-testable without the
heavy detection environment, mirroring ``distance.py``.
"""

__all__ = ["compute_descriptor", "descriptor_distance", "AppearanceStore", "blob_contrast"]

import numpy as np

# Histogram resolution.  Small on purpose: drone/insect motion blobs are tiny,
# so a coarse histogram is both cheaper and more robust to a few stray pixels.
_HUE_BINS = 8     # HSV hue bins   (3-channel frames)
_SAT_BINS = 4     # HSV saturation bins
_GRAY_BINS = 16   # intensity bins (1-channel / thermal frames)


# ---------------------------------------------------------------------------
# Descriptor extraction (cv2 used locally)
# ---------------------------------------------------------------------------

def compute_descriptor(frame, box, mask=None, min_box_px: int = 3, min_pixels: int = 8):
    """Return a normalised appearance histogram for the patch under ``box``.

    Args:
        frame:      BGR (H, W, 3) or single-channel (H, W) image.  This should be
                    the frame the motion mask is *aligned to*.  The frame-diff
                    detector fires on motion between the previous and current
                    frame (with temporal accumulation), so the blob trails the
                    object by ~1 frame — pass the **previous** frame here so the
                    sampled pixels are the object, not its trailing background.
        box:        (x1, y1, x2, y2) bounding rect (ints, may exceed bounds).
        mask:       Optional full-frame uint8 motion mask (255 = motion).  When
                    given, the histogram is built **only over the motion pixels**
                    inside ``box`` — this excludes the surrounding sky/background
                    that otherwise dominates a small object's bounding box, so
                    two objects against the same backdrop stay distinguishable.
        min_box_px: Minimum patch width/height; smaller patches return ``None``
                    (too few pixels for a meaningful histogram).
        min_pixels: Minimum number of motion pixels (when ``mask`` is given) for
                    a *reliable* descriptor.  Below this the histogram is too
                    noisy to compare frame-to-frame (measured: <8 px objects have
                    a median self-distance of ~0.27 vs ~0.0 for larger ones), and
                    such tiny objects also can't be told apart by appearance, so
                    we return ``None`` and let the tracker fall back to
                    position + size.  Ignored when ``mask`` is None.

    Returns:
        ``float32`` 1-D histogram L1-normalised to sum 1, or ``None`` if the
        box is degenerate, the patch is too small, too few motion pixels, or no
        pixels were counted.
    """
    import cv2  # local import — keeps module top-level numpy-only

    if frame is None or box is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    # Clamp to frame bounds (boxes come from contours on a possibly-masked frame).
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
        return None

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    mask_patch = None
    if mask is not None:
        mask_patch = mask[y1:y2, x1:x2]
        if mask_patch.dtype != np.uint8:
            mask_patch = mask_patch.astype(np.uint8)
        if not mask_patch.flags["C_CONTIGUOUS"]:
            mask_patch = np.ascontiguousarray(mask_patch)
        # Too few object pixels → descriptor is unreliable (and useless for
        # tiny objects, whose appearances aren't separable anyway).
        if int(cv2.countNonZero(mask_patch)) < min_pixels:
            return None

    if patch.ndim == 3 and patch.shape[2] == 3:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], mask_patch,
            [_HUE_BINS, _SAT_BINS], [0, 180, 0, 256],
        )
    else:
        gray = patch if patch.ndim == 2 else patch[:, :, 0]
        hist = cv2.calcHist([gray], [0], mask_patch, [_GRAY_BINS], [0, 256])

    hist = hist.astype(np.float32).flatten()
    total = float(hist.sum())
    if total <= 0.0:
        return None
    hist /= total
    return hist


# ---------------------------------------------------------------------------
# Descriptor comparison (pure numpy)
# ---------------------------------------------------------------------------

def descriptor_distance(a, b):
    """Bhattacharyya distance between two L1-normalised histograms.

    Returns a value in ``[0, 1]`` — 0 for identical distributions, approaching
    1 for fully disjoint ones.  Returns ``None`` if either input is ``None`` or
    the shapes differ (caller treats ``None`` as "no appearance signal").
    """
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    bc = float(np.sum(np.sqrt(a * b)))   # Bhattacharyya coefficient ∈ [0, 1]
    return float(np.sqrt(max(0.0, 1.0 - bc)))


# ---------------------------------------------------------------------------
# Per-track template store
# ---------------------------------------------------------------------------

_DORMANT_UNKNOWN = 10 ** 9   # frames_since_match for an unknown track


class AppearanceStore:
    """Per-track appearance template + maturity/dormancy state.

    Updated once per frame (single-threaded, right after ``tracker.update()``)
    from the returned active objects.  For each track it keeps:

    * ``templates[id]`` — EMA of the descriptors from frames the track was
      *actually matched* (a small ``alpha`` keeps a transient ambiguous frame
      from overwriting it).  Crucially the template is **not** re-blended while a
      track merely coasts, so a mature object's template stays clean for
      re-identification when it resumes after a stop.
    * ``match_count`` — number of frames the track was matched (its *maturity*).
    * ``last_matched_tick`` — last frame it was matched (its *dormancy*).

    A frame counts as a *match* when the track's Norfair ``hit_counter`` did not
    decrease since the previous frame (Norfair adds +2 on a hit, subtracts 1 on a
    miss, and caps at ``hit_counter_max``), so ``cur >= prev`` ⇔ matched.  This
    is robust to ``id()`` reuse of Detection objects.

    Args:
        alpha:       EMA blend weight for new descriptors (0 < alpha <= 1).
        prune_after: Drop a track's state after this many consecutive ``update``
                     calls in which it was absent (caps memory on long runs).
                     Must exceed the ReID window so a dead mature track's
                     template survives long enough to be reclaimed.  <=0 disables.
    """

    def __init__(self, alpha: float = 0.15, prune_after: int = 1200):
        self.alpha       = float(alpha)
        self.prune_after = int(prune_after)
        self.templates   = {}        # track_id -> float32 histogram
        self._last_seen  = {}        # track_id -> update tick when last present
        self._match_count = {}       # track_id -> # frames matched (maturity)
        self._last_matched_tick = {} # track_id -> tick of last actual match
        self._prev_hit   = {}        # track_id -> hit_counter at previous update
        self._tick       = 0

    def update(self, tracked_objects) -> None:
        """Record per-track state and blend templates for matched tracks."""
        self._tick += 1
        for obj in tracked_objects:
            tid = obj.id
            self._last_seen[tid] = self._tick

            # Matched this frame?  hit_counter did not drop (hit: +2 / miss: -1).
            cur_hit = getattr(obj, "hit_counter", None)
            prev_hit = self._prev_hit.get(tid)
            if cur_hit is None:
                matched = True                      # no signal → assume matched
            elif prev_hit is None:
                matched = True                      # first sighting = a hit
            else:
                matched = cur_hit >= prev_hit
            if cur_hit is not None:
                self._prev_hit[tid] = cur_hit

            if not matched:
                continue                            # coasting → keep state frozen

            self._match_count[tid] = self._match_count.get(tid, 0) + 1
            self._last_matched_tick[tid] = self._tick

            last_det = getattr(obj, "last_detection", None)
            emb = getattr(last_det, "embedding", None) if last_det is not None else None
            if emb is None:
                continue                            # matched but no usable descriptor
            emb = np.asarray(emb, dtype=np.float32)

            prev = self.templates.get(tid)
            if prev is None or prev.shape != emb.shape:
                self.templates[tid] = emb.copy()    # seed on first descriptor
            else:
                a = self.alpha
                self.templates[tid] = (1.0 - a) * prev + a * emb

        self._prune()

    def get(self, track_id):
        """Return the template for ``track_id`` (or ``None`` if unknown)."""
        return self.templates.get(track_id)

    def is_mature(self, track_id, maturity_frames) -> bool:
        """True if the track has been matched on >= ``maturity_frames`` frames."""
        return self._match_count.get(track_id, 0) >= maturity_frames

    def frames_since_match(self, track_id) -> int:
        """Frames since the track was last actually matched (large if unknown)."""
        t = self._last_matched_tick.get(track_id)
        if t is None:
            return _DORMANT_UNKNOWN
        return self._tick - t

    def clear(self) -> None:
        """Forget all state (call between videos in batch mode)."""
        self.templates.clear()
        self._last_seen.clear()
        self._match_count.clear()
        self._last_matched_tick.clear()
        self._prev_hit.clear()
        self._tick = 0

    def _prune(self) -> None:
        if self.prune_after <= 0:
            return
        stale = [
            tid for tid, seen in self._last_seen.items()
            if self._tick - seen > self.prune_after
        ]
        for tid in stale:
            self.templates.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._match_count.pop(tid, None)
            self._last_matched_tick.pop(tid, None)
            self._prev_hit.pop(tid, None)


# ---------------------------------------------------------------------------
# Blob-vs-background contrast (drift / sky-track rejection)
# ---------------------------------------------------------------------------

def blob_contrast(frame, box, mask, pad: int = 8, min_obj: int = 2, min_bg: int = 5):
    """Intensity contrast of a detection's blob against its local background.

    A real object (a dark/bright speck against sky) stands out: the median
    intensity of its **motion-mask pixels** differs strongly from the median of
    the surrounding non-motion (sky) pixels.  A spurious "drifting" detection
    that is really just sky / warp residual has near-zero contrast — its masked
    pixels look the same as the background.  Used by the track filter to drop
    sky-like drift tracks.

    Args:
        frame: BGR or single-channel image the ``mask`` is aligned to.
        box:   (x1, y1, x2, y2) detection bounding rect.
        mask:  full-frame uint8 motion mask (255 = motion).
        pad:   px to expand the box by to sample local background (sky) pixels.

    Returns:
        ``abs(median(object) - median(background))`` (float), or ``None`` if
        there are too few object or background pixels to measure.
    """
    import cv2  # local import — keeps module top-level numpy-only

    if frame is None or box is None or mask is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if (frame.ndim == 3 and frame.shape[2] == 3) \
        else (frame if frame.ndim == 2 else frame[:, :, 0])
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    op = gray[y1:y2, x1:x2]; mp = mask[y1:y2, x1:x2]
    obj = op[mp > 0]
    outer = gray[oy1:oy2, ox1:ox2]; outm = mask[oy1:oy2, ox1:ox2]
    bg = outer[outm == 0]
    if obj.size < min_obj or bg.size < min_bg:
        return None
    return abs(float(np.median(obj)) - float(np.median(bg)))
