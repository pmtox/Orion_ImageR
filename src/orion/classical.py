from __future__ import annotations

import cv2
import numpy as np


def inpaint_telea(damaged_rgb: np.ndarray, mask: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """Run OpenCV Telea inpainting; any positive mask value is inpainted."""
    if damaged_rgb.ndim != 3 or damaged_rgb.shape[-1] != 3 or damaged_rgb.dtype != np.uint8:
        raise ValueError("damaged_rgb must be an HxWx3 RGB uint8 array")
    if mask.shape != damaged_rgb.shape[:2]:
        raise ValueError("mask spatial shape must match image")
    if radius <= 0:
        raise ValueError("radius must be positive")
    binary = (mask > 0).astype(np.uint8) * 255
    # OpenCV accepts BGR images; channel permutation has no effect on the algorithm.
    bgr = cv2.cvtColor(damaged_rgb, cv2.COLOR_RGB2BGR)
    output = cv2.inpaint(bgr, binary, float(radius), cv2.INPAINT_TELEA)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
