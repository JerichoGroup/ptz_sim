"""Simple image augmentation for training data preparation."""

__all__ = ["augment"]

import cv2
import numpy as np


def augment(img):
    """Apply blur, Gaussian noise, and random brightness to ``img``.

    Args:
        img: uint8 numpy array (H, W, C) or (H, W).

    Returns:
        Augmented uint8 array of the same shape.
    """
    img   = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.normal(0, 5, img.shape)
    img   = np.clip(img + noise, 0, 255).astype(np.uint8)
    alpha = np.random.uniform(0.7, 1.3)
    return np.clip(img * alpha, 0, 255).astype(np.uint8)
