from __future__ import annotations

import torch
from torch import nn


class CNNInpaintingBaseline(nn.Module):
    """Small mask-conditioned CNN; architecture scaffold, not trained weights."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, damaged_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inputs: NCHW float tensors in [0,1]; output has RGB shape and range."""
        if damaged_image.ndim != 4 or damaged_image.shape[1] != 3:
            raise ValueError("damaged_image must have shape (B,3,H,W)")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape (B,1,H,W)")
        if damaged_image.shape[0] != mask.shape[0] or damaged_image.shape[2:] != mask.shape[2:]:
            raise ValueError("image and mask batch/spatial dimensions must match")
        return self.net(torch.cat([damaged_image, mask], dim=1))
