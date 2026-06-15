"""Online ego-motion calibration for the PTZ tracking pipeline.

The live pipeline learns ``k_pan`` and ``k_tilt`` online — the number of
camera-image pixels produced per ONVIF pose unit per zoom-magnification unit —
via an Exponential Moving Average (EMA) over measured (affine-translation,
pose-delta) pairs.

The core arithmetic is extracted here as pure Python/numpy so it can be tested
without cv2, norfair, or an ONVIF camera.
"""

__all__ = ["update_ego_gain"]


def update_ego_gain(
    prev_k:     float,
    translation_px: float,
    pose_delta: float,
    zoom_mag:   float,
    ema:        float,
    min_dp:     float = 1e-6,
) -> float:
    """Update one axis of the ego-motion calibration gain via EMA.

    The gain ``k`` represents:  image_translation_px = k * pose_delta * zoom_mag

    A single new measurement is:  k_meas = translation_px / (pose_delta * zoom_mag)
    The EMA update is:            k_new  = ema * k_prev + (1 - ema) * k_meas

    Args:
        prev_k:         Previous gain estimate for this axis (pan or tilt).
        translation_px: Affine translation component in pixels (from
                        ``mt.last_affine[0, 2]`` for pan or ``[1, 2]`` for tilt).
        pose_delta:     Change in ONVIF pan/tilt position since the last frame.
        zoom_mag:       Current optical magnification (from ``_zoom_mag(zoom_pos)``).
        ema:            EMA decay coefficient (0 < ema < 1; typical: 0.95).
        min_dp:         Minimum ``|pose_delta|`` to accept a measurement — guards
                        against division by near-zero deltas producing spurious gains.
                        Default is a tiny epsilon; the pipeline uses CALIB_MIN_DP
                        (typically 0.001–0.01 ONVIF units).

    Returns:
        Updated gain estimate, or ``prev_k`` unchanged if the pose delta is too
        small to yield a reliable measurement (i.e. ``|pose_delta| <= min_dp``).
    """
    if abs(pose_delta) <= min_dp or zoom_mag == 0.0:
        return prev_k
    k_measured = translation_px / (pose_delta * zoom_mag)
    return ema * prev_k + (1.0 - ema) * k_measured
