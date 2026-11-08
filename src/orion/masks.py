from __future__ import annotations

import numpy as np


def make_rect_mask(
    height: int,
    width: int,
    *,
    x: int,
    y: int,
    rect_width: int,
    rect_height: int,
    soft: bool = False,
    edge_width: int = 8,
) -> np.ndarray:
    """Create an HxW float32 damage mask (0 visible, 1 damaged).

    Soft masks taper linearly at the rectangle boundary; the central region is 1.
    This is a graded synthetic boundary, not a probabilistic damage estimate.
    """
    if min(height, width, rect_width, rect_height) <= 0:
        raise ValueError("Image and rectangle dimensions must be positive")
    if edge_width <= 0:
        raise ValueError("edge_width must be positive")
    x0, y0 = int(np.clip(x, 0, width)), int(np.clip(y, 0, height))
    x1, y1 = int(np.clip(x + rect_width, 0, width)), int(np.clip(y + rect_height, 0, height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Rectangle does not overlap the image")
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    if not soft:
        return mask

    # Distance to nearest edge within the rectangle; ramp from 0 to 1.
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.minimum.reduce([xx - x0 + 1, x1 - xx, yy - y0 + 1, y1 - yy]).astype(np.float32)
    mask[y0:y1, x0:x1] = np.clip(distance / float(edge_width), 0.0, 1.0)
    return mask
