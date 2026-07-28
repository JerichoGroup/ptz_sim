import cv2
import numpy as np


# Scale at which corners are *detected* (cheap full-image scan).
# LK tracking and affine estimation always run at full resolution so there
# is no translation rescaling and no error amplification.
STAB_SCALE = 0.25

# Camera-move detection: translation magnitude above this (px/frame) triggers a move.
# Static cameras measure ~0.01 px; PTZ slews measure 0.2–35 px.
BIG_MOVE_PX = 0.2

# Adaptive resume: resume detection when mean|diff| <= baseline * this factor.
# The baseline is learned from normal (non-moving) frames via EMA.
SETTLE_RESIDUAL_FACTOR = 1.5

# Hard cap on settle frames — prevents infinite suppression if residual never calms.
MAX_SETTLE_FRAMES = 30

# EMA smoothing for the learned static-scene baseline diff.
# 0.95 → slow to update, 0.5 → fast; slow is correct so one noisy frame doesn't inflate it.
BASELINE_EMA = 0.95

# When the camera is known-static, skip the full stabilization pipeline if the
# raw frame diff mean is below this threshold (sensor noise is ~0.5–2 px units;
# any real camera movement produces a much larger mean).
STATIC_DIFF_THRESHOLD = 0.2

# Per-frame scale change (from the similarity matrix) that triggers a camera move.
# Pure-zoom maneuvers have near-zero translation; this catches them.
BIG_ZOOM = 0.01

# Pixels with raw (unwarped) frame-diff below this are treated as image-fixed
# overlays (OSD timestamp etc.) and suppressed from the stabilized diff.
# A fixed overlay has near-zero raw diff (~4–5 on this camera) while the
# background warps under it, creating a large spurious stabilized diff.
OVERLAY_RAW_TAU = 12

# --- Move-frame detection: emit motion detections during a camera slew ---

# Accumulation fire threshold DURING a move (vs static 1.5). A compact drone
# re-hits the same world cell faster than diffuse warp residual; raise the bar
# to buy selectivity. Tune down toward 1.5 if drones are missed.
MOVE_ACCUM_THRESH = 2.5

# Per-frame threshold = max(mean + MOVE_STD_MULT*std, MOVE_DIFF_FLOOR) during a
# move (static uses multiplier 3.0). The residual histogram during a slew is
# heavy-tailed (horizon/cloud edges), so 3*std under-thresholds.
MOVE_STD_MULT = 4.0

# Absolute intensity floor for the move threshold. Guards the case where
# mean+k*std collapses momentarily mid-slew; real drone contrast is well above.
MOVE_DIFF_FLOOR = 18

# Largest connected blob (px) allowed through the move mask. Drones are 2–80 px;
# horizon/cloud/parallax residual forms blobs of 15k–22k px (observed). Generous
# headroom above 80 absorbs warp-smear inflation.
MOVE_MAX_BLOB_AREA = 400


class MotionDetector:
    WINDOW_NAME = "motion"
    SHOW_WINDOW = False

    def __init__(
        self,
        min_area=2,
        max_area=80,
        accumulation_decay=0.92,
        trajectory_length=10,
        consistency_threshold=3,
        ground_mask_generator=None,
    ):
        self.ground_mask_generator = ground_mask_generator
        if self.SHOW_WINDOW:
            cv2.namedWindow(MotionDetector.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(MotionDetector.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.prev_gray = None
        self.motion_history = None  # preallocated float32 on first frame
        self._thresh_f32 = None    # preallocated buffers for temporal_accumulation
        self._mask_bool  = None
        self._mask_out   = None

        self.min_area = min_area
        self.max_area = max_area
        self.accumulation_decay = accumulation_decay

        self.tracks = {}
        self.trajectory_length = trajectory_length
        self.consistency_threshold = consistency_threshold

        # Camera-move state (readable by callers).
        self.camera_moving = False
        self.settle_frames  = 0          # frames spent settling since last move ended
        self.baseline_diff  = None       # learned mean|diff| for the static scene
        self.last_affine    = None       # most recent 2×3 affine (prev→curr), None on failure

    @staticmethod
    def stabilize_frame(prev_gray, gray):
        """Estimate the camera-motion affine and warp prev_gray to align with gray.

        Returns (stabilized_frame, motion_magnitude_px).
        motion_magnitude_px is None on any estimation failure (treated as a big move).

        Corner detection runs on a cheap downscaled copy of gray (avoids a full
        4.9 MP scan). The detected points are snapped to full-resolution gradient
        peaks via cornerSubPix, then LK + affine + warp run at full resolution —
        no translation rescaling, no error amplification.
        """
        # Cheap corner detection on downscaled image.
        small_gray = cv2.resize(gray, (0, 0), fx=STAB_SCALE, fy=STAB_SCALE,
                                interpolation=cv2.INTER_LINEAR)
        curr_pts = cv2.goodFeaturesToTrack(
            small_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10
        )

        if curr_pts is None:
            return gray, None, None

        # Scale corner coordinates back to full-resolution, then snap each
        # point to its nearest real corner at full res (cheap — ~200 small
        # windows). Without this, scaled-up positions often miss actual
        # gradient peaks and LK tracking fails.
        curr_pts = (curr_pts / STAB_SCALE).astype(np.float32)
        curr_pts = cv2.cornerSubPix(
            gray, curr_pts,
            winSize=(5, 5), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
        )

        # LK tracking at full resolution — only samples ~200 small windows,
        # not the whole image, so this is fast.
        prev_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            prev_gray,
            curr_pts,
            None
        )

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 6:
            return gray, None, None

        # Affine estimation at full resolution — no translation rescaling needed.
        matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)

        if matrix is None:
            return gray, None, None

        motion_mag = float(np.hypot(matrix[0, 2], matrix[1, 2]))

        stabilized = cv2.warpAffine(
            prev_gray,
            matrix,
            (prev_gray.shape[1], prev_gray.shape[0])
        )

        return stabilized, motion_mag, matrix

    def temporal_accumulation(self, thresh):
        np.multiply(self.motion_history, self.accumulation_decay, out=self.motion_history)
        np.multiply(thresh, np.float32(1.0 / 255.0), out=self._thresh_f32)
        np.add(self.motion_history, self._thresh_f32, out=self.motion_history)
        np.greater(self.motion_history, np.float32(1.5), out=self._mask_bool)
        np.multiply(self._mask_bool, np.uint8(255), out=self._mask_out)
        return self._mask_out

    @staticmethod
    def _suppress_large_blobs(mask, max_area):
        """Zero any connected component whose pixel area exceeds max_area.

        The decisive cheap filter against large horizon/cloud/parallax residual
        blobs (observed 15k–22k px) that survive the threshold step during a slew.
        Drones are 2–80 px; MOVE_MAX_BLOB_AREA gives generous headroom for
        warp-smear inflation while still being far below residual blob sizes.
        """
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = mask.copy()
        for lbl in range(1, stats.shape[0]):   # skip background label 0
            if stats[lbl, cv2.CC_STAT_AREA] > max_area:
                out[labels == lbl] = 0
        return out

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.motion_history = np.zeros(gray.shape, dtype=np.float32)
            self._thresh_f32 = np.empty(gray.shape, dtype=np.float32)
            self._mask_bool  = np.empty(gray.shape, dtype=bool)
            self._mask_out   = np.empty(gray.shape, dtype=np.uint8)
            return np.zeros_like(gray)

        # --- Stabilization ---
        # Always stabilize during a slew (need the affine to warp the history buffer).
        # When static, take the cheap path first and skip stabilization entirely if
        # the raw diff is tiny (sensor noise level).
        raw_diff = cv2.absdiff(self.prev_gray, gray)
        if self.camera_moving:
            stabilized, motion_mag, matrix = self.stabilize_frame(self.prev_gray, gray)
            diff = cv2.absdiff(stabilized, gray)
        else:
            if float(cv2.mean(raw_diff)[0]) < STATIC_DIFF_THRESHOLD:
                diff = raw_diff
                motion_mag, matrix = 0.0, None
            else:
                stabilized, motion_mag, matrix = self.stabilize_frame(self.prev_gray, gray)
                diff = cv2.absdiff(stabilized, gray)
        self.last_affine = matrix

        # --- Camera-move flag (computed early: only needs motion_mag + scale) ---
        # Scale from the similarity matrix catches near-stationary zooms that have
        # tiny translation and wouldn't trigger BIG_MOVE_PX alone.
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0])) if matrix is not None else 1.0
        is_big_move = (motion_mag is None) or (motion_mag > BIG_MOVE_PX) \
                      or (abs(scale - 1.0) > BIG_ZOOM)

        # --- Clean the stabilized diff ---
        # Only applies when we warped (matrix is not None). Two sources of artifact:
        #   Borders: warpAffine fills uncovered regions with 0; absdiff sees the full
        #     current-frame intensity there — looks like motion, isn't.
        #   OSD overlay: fixed on-screen elements (timestamp etc.) never change between
        #     frames (raw diff ≈4–5) but the background is warped under them → large
        #     spurious stabilized diff. Suppress where raw diff is near zero.
        if matrix is not None:
            h_g, w_g = gray.shape
            valid = cv2.warpAffine(
                np.ones((h_g, w_g), dtype=np.uint8),
                matrix, (w_g, h_g), borderValue=0
            )
            valid = cv2.erode(valid, np.ones((3, 3), dtype=np.uint8))
            diff[valid == 0] = 0
            diff[raw_diff < OVERLAY_RAW_TAU] = 0

        # --- Ground mask: warp it forward during a move, then apply ---
        # Warp-from-anchor keeps the horizon masked as the camera slews so shifted
        # horizon edges don't become false motion. The warp happens before diff is
        # consumed so the mask aligns to the current frame (no one-frame lag).
        if self.ground_mask_generator is not None:
            if is_big_move:
                self.ground_mask_generator.compute(frame)
            mask = self.ground_mask_generator.get_mask()
            if mask is not None:
                diff[mask > 0] = 0

        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        mean_val, std_val = cv2.meanStdDev(diff)
        mean_val = float(mean_val[0, 0])
        std_val  = float(std_val[0, 0])
        _, max_val, _, _ = cv2.minMaxLoc(diff)

        if is_big_move:
            # Warp+decay the history buffer so the drone's accumulated signal stays
            # aligned and fades gracefully: a short move keeps it; a long move clears it.
            self.camera_moving = True
            self.settle_frames = 0
            h, w = gray.shape
            if matrix is not None:
                self.motion_history = cv2.warpAffine(
                    self.motion_history, matrix, (w, h), borderValue=0.0)
            else:
                self.motion_history[:] = 0  # no affine → can't align, flush instead
            self.motion_history *= self.accumulation_decay

            if matrix is None:
                # No valid affine: can't trust the diff or align history — emit nothing.
                self.prev_gray = gray
                return np.zeros_like(gray)

            # --- Emit detections during the slew ---
            # Run the already-cleaned diff through a move-aware threshold + accumulation.
            # Discriminators vs warp residual:
            #   1. Ground mask was warped above → horizon/cloud edges are still masked.
            #   2. Temporal accumulation at MOVE_ACCUM_THRESH (> static 1.5): a compact
            #      drone re-hits the same world cell faster than diffuse residual.
            #   3. MOVE_MAX_BLOB_AREA cap: horizon/parallax residual forms blobs orders of
            #      magnitude larger than a 2–80 px drone.
            # Note: history was already decayed above (step A); only add here (no double
            # decay). Fire at MOVE_ACCUM_THRESH rather than calling temporal_accumulation().
            move_thresh_val = max(mean_val + MOVE_STD_MULT * std_val, MOVE_DIFF_FLOOR)
            _, thresh = cv2.threshold(diff, move_thresh_val, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            self.motion_history += thresh.astype(np.float32) / 255.0
            move_mask = (self.motion_history > MOVE_ACCUM_THRESH).astype(np.uint8) * 255
            move_mask = self._suppress_large_blobs(move_mask, MOVE_MAX_BLOB_AREA)
            self.prev_gray = gray
            if self.SHOW_WINDOW:
                cv2.imshow(MotionDetector.WINDOW_NAME, move_mask)
                cv2.waitKey(1)
            return move_mask

        # --- Adaptive settle: ringing window after the move stops ---
        if self.camera_moving:
            self.settle_frames += 1
            base = self.baseline_diff if self.baseline_diff is not None else mean_val
            settled = (mean_val <= base * SETTLE_RESIDUAL_FACTOR) \
                      or (self.settle_frames >= MAX_SETTLE_FRAMES)
            if not settled:
                self.motion_history[:] = 0
                self.prev_gray = gray
                return np.zeros_like(gray)
            # Residual back to normal: flip the flag (True→False edge lets the caller
            # recompute the ground mask) and fall through to the first real frame.
            self.camera_moving = False

        # --- Normal detection pipeline ---

        if self.baseline_diff is None:
            self.baseline_diff = mean_val
        else:
            self.baseline_diff = BASELINE_EMA * self.baseline_diff \
                                 + (1.0 - BASELINE_EMA) * mean_val

        threshold = 5 if (max_val / mean_val > 5 if mean_val > 0 else False) \
                      else mean_val + 3 * std_val
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        accumulation_mask = self.temporal_accumulation(thresh)

        self.prev_gray = gray

        if self.SHOW_WINDOW:
            cv2.imshow(MotionDetector.WINDOW_NAME, accumulation_mask)
            if cv2.waitKey(1) == 27:
                return accumulation_mask

        return accumulation_mask
