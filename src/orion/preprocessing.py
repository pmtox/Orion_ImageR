from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(source: str | Path | bytes | BytesIO, max_side: int | None = None) -> np.ndarray:
    """Load an image as RGB uint8 and optionally downscale its longest side."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if max_side is not None:
            if max_side <= 0:
                raise ValueError("max_side must be positive")
            scale = min(1.0, max_side / max(image.size))
            if scale < 1:
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).copy()


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 [0,255] to float32 [0,1]."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    return image.astype(np.float32) / 255.0
