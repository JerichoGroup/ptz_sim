"""Template-at-location tracking for near-stationary objects.

For a track that is *mature* and moving *very slowly* (≈ stationary), the motion
detector often produces nothing (no movement → no frame-diff blob), so the
tracker either coasts/dies or a passing object's motion blob hijacks its ID.
For such a track the right question is not "which motion blob is nearest?" but
"is my object still right here?" — answered by template-matching the object's
own stored appearance patch at its predicted location.

``TemplateTracker`` maintains a small grayscale **reference patch** per track id:

* ``observe(frame, tracked_objects)`` — after ``tracker.update()`` — refreshes
  each track's patch from its latest *real* (non-synthetic) detection.
* ``augment(frame, prev_tracked, motion_dets)`` — before ``tracker.update()`` —
  for every mature, slow track it ``cv2.matchTemplate``-searches a small window
  around the track's estimate.  If the blob is still there (peak correlation ≥
  threshold) it emits a **synthetic** ``Detection`` at the matched location
  (pin + re-localize) and suppresses any motion detection co-located with it, so
  the slow object is represented once — by its own appearance, not by whatever
  motion happened to be nearby.  If the blob has changed/gone it emits nothing
  and the track is left to coast (the object moved away → re-acquired normally).

cv2 is imported lazily inside the methods so the module top stays numpy-only
(consistent with the other ``dronetracker.tracking`` modules / unit-testable).
"""

__all__ = ["TemplateTracker"]

import numpy as np

from dronetracker.tracking.appearance import descriptor_distance


class TemplateTracker:
    """Per-track reference-patch store + template-match augment/observe hooks.

    Args:
        max_speed:        Only run for tracks whose Kalman speed is below this
                          (px/frame) — i.e. essentially stationary.
        search_radius:    Half-size (px) of the search window around the
                          estimate in which the patch is matched.
        match_threshold:  Minimum ``TM_CCOEFF_NORMED`` peak to accept "blob
                          present" at the location.
        pad:              Pixels added around the detection box to form the
                          reference patch (gives the template some context).
        suppress_radius:  Motion detections within this many px of a synthetic
                          detection are dropped (avoid a duplicate/phantom).
        maturity_frames:  Real observations required before a track is eligible
                          (well-established appearance).
        min_patch:        Minimum patch side (px); smaller boxes are padded up.
        appearance_store: Optional ``AppearanceStore`` — used to tell the
                          object's OWN motion (appearance matches its template →
                          let it drive the track) from a different intruder.
        match_dist:       Max Bhattacharyya distance for a nearby motion's
                          embedding to count as "the object" (vs an intruder).
    """

    def __init__(
        self,
        max_speed: float = 1.0,
        search_radius: int = 20,
        match_threshold: float = 0.5,
        pad: int = 4,
        suppress_radius: int = 10,
        maturity_frames: int = 45,
        min_patch: int = 7,
        appearance_store=None,
        match_dist: float = 0.6,
    ):
        self.max_speed       = float(max_speed)
        self.search_radius    = int(search_radius)
        self.match_threshold  = float(match_threshold)
        self.pad              = int(pad)
        self.suppress_radius  = int(suppress_radius)
        self.maturity_frames  = int(maturity_frames)
        self.min_patch        = int(min_patch)
        self.appearance_store = appearance_store
        self.match_dist       = float(match_dist)
        self._patch = {}   # track_id -> uint8 grayscale reference patch
        self._seen  = {}   # track_id -> count of real observations (maturity)

    # ── lifecycle ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Forget all patches (call between videos in batch mode)."""
        self._patch.clear()
        self._seen.clear()

    # ── post-update: refresh reference patches from REAL detections ──────────

    def observe(self, frame, tracked_objects) -> None:
        import cv2
        gray = self._gray(frame, cv2)
        for obj in tracked_objects:
            ld = getattr(obj, "last_detection", None)
            if ld is None or getattr(ld, "synthetic", False):
                continue                          # no detection, or our own synthetic
            box = getattr(ld, "box", None)
            if box is None:
                continue
            patch = self._extract(gray, box)
            if patch is None:
                continue
            self._seen[obj.id] = self._seen.get(obj.id, 0) + 1
            self._patch[obj.id] = patch           # always reflect the latest real sighting

    # ── pre-update: synthesize detections for mature, slow tracks ────────────

    def augment(self, frame, prev_tracked, motion_dets):
        """Return ``(extra_detections, drop_indices)``.

        ``extra_detections`` are appended to the detection list; ``drop_indices``
        are indices into ``motion_dets`` to remove (suppressed near a synthetic).

        Appearance-gated hand-off: if a motion detection near a slow track's
        estimate *matches the track's appearance template*, it is the object
        itself (creeping or resuming) — we emit nothing and suppress nothing, so
        the object's OWN motion drives the track (same ID, no velocity spike).
        Only when there is no such object-motion do we template-match and pin:
        blob present → synthetic detection + suppress the appearance-different
        motion (an intruder); blob gone → nothing (released).
        """
        if not prev_tracked:
            return [], set()
        import cv2
        from norfair import Detection

        gray = self._gray(frame, cv2)
        sr2 = self.search_radius * self.search_radius
        extra, drop = [], set()
        for obj in prev_tracked:
            tid = getattr(obj, "id", None)
            if tid is None or self._seen.get(tid, 0) < self.maturity_frames:
                continue
            patch = self._patch.get(tid)
            if patch is None:
                continue
            speed = self._speed(obj)
            if speed is None or speed >= self.max_speed:
                continue                          # not (reliably) slow → skip
            cx, cy = float(obj.estimate[0][0]), float(obj.estimate[0][1])

            # Motion detections near the estimate, and whether any is the object
            # itself (appearance matches the track's template).
            template = self.appearance_store.get(tid) if self.appearance_store is not None else None
            nearby = []                           # (index, md)
            object_motion_present = False
            for i, md in enumerate(motion_dets):
                mdx, mdy = float(md.points[0][0]), float(md.points[0][1])
                if (mdx - cx) ** 2 + (mdy - cy) ** 2 > sr2:
                    continue
                nearby.append(i)
                emb = getattr(md, "embedding", None)
                if template is not None and emb is not None:
                    d = descriptor_distance(emb, template)
                    if d is not None and d <= self.match_dist:
                        object_motion_present = True
                        break
                else:
                    # No appearance signal → can't distinguish; treat nearby
                    # motion as the object (gap-filler) and let it drive.
                    object_motion_present = True
                    break
            if object_motion_present:
                continue                          # object's own motion drives the track

            # No object-motion nearby → verify the blob is still there and pin.
            hit = self._match(gray, patch, cx, cy, cv2)
            if hit is None:
                continue
            score, mx, my = hit
            if score < self.match_threshold:
                continue                          # blob changed / gone → emit nothing
            ph, pw = patch.shape[:2]
            det = Detection(points=np.array([[mx, my]], dtype=float))
            det.area = float(pw * ph)
            det.box = (int(mx - pw / 2), int(my - ph / 2),
                       int(mx + pw / 2), int(my + ph / 2))
            det.embedding = None
            det.synthetic = True
            extra.append(det)
            r2 = self.suppress_radius * self.suppress_radius
            for i in nearby:
                md = motion_dets[i]
                mdx, mdy = float(md.points[0][0]), float(md.points[0][1])
                if (mdx - mx) ** 2 + (mdy - my) ** 2 <= r2:
                    drop.add(i)                   # appearance-different intruder
        return extra, drop

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _gray(frame, cv2):
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame if frame.ndim == 2 else frame[:, :, 0]

    def _extract(self, gray, box):
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        # pad, and enforce a minimum patch side
        x1 -= self.pad; y1 -= self.pad; x2 += self.pad; y2 += self.pad
        if x2 - x1 < self.min_patch:
            cxm = (x1 + x2) // 2
            x1, x2 = cxm - self.min_patch // 2, cxm + (self.min_patch - self.min_patch // 2)
        if y2 - y1 < self.min_patch:
            cym = (y1 + y2) // 2
            y1, y2 = cym - self.min_patch // 2, cym + (self.min_patch - self.min_patch // 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < self.min_patch or y2 - y1 < self.min_patch:
            return None
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0 or float(patch.std()) < 1e-3:
            return None                           # flat patch → matchTemplate undefined
        return np.ascontiguousarray(patch)

    def _speed(self, obj):
        vel = getattr(obj, "estimate_velocity", None)
        if vel is None:
            return None
        v = np.asarray(vel, dtype=float).ravel()
        if v.size < 2:
            return None
        return float(np.hypot(v[0], v[1]))

    def _match(self, gray, patch, cx, cy, cv2):
        """Match ``patch`` in a window around (cx, cy); return (score, mx, my)."""
        h, w = gray.shape[:2]
        ph, pw = patch.shape[:2]
        half_w = self.search_radius + pw // 2
        half_h = self.search_radius + ph // 2
        x0 = max(0, int(cx - half_w)); x1 = min(w, int(cx + half_w))
        y0 = max(0, int(cy - half_h)); y1 = min(h, int(cy + half_h))
        win = gray[y0:y1, x0:x1]
        if win.shape[0] < ph or win.shape[1] < pw:
            return None                           # window smaller than patch
        if float(win.std()) < 1e-3:
            return None
        res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        mx = x0 + max_loc[0] + pw / 2.0
        my = y0 + max_loc[1] + ph / 2.0
        return float(max_val), mx, my
