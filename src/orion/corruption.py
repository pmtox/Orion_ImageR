from __future__ import annotations

import numpy as np


def apply_corruption(image: np.ndarray, mask: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """Blend deterministic random-color corruption into RGB uint8 image by mask."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 RGB uint8 array")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask spatial shape must match image")
    if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("mask values must be finite and in [0,1]")
    rng = np.random.default_rng(seed)
    color = rng.integers(0, 256, size=(1, 1, 3), dtype=np.uint8).astype(np.float32)
    alpha = mask.astype(np.float32)[..., None]
    result = image.astype(np.float32) * (1.0 - alpha) + color * alpha
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
