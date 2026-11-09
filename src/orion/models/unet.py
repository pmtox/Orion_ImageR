from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNetInpaintingModel(nn.Module):
    """Compact U-Net that takes RGB damaged image + one-channel mask."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(4, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, 3, kernel_size=1)

    @staticmethod
    def _up_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, damaged_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if damaged_image.ndim != 4 or damaged_image.shape[1] != 3:
            raise ValueError("damaged_image must have shape (B,3,H,W)")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape (B,1,H,W)")
        if damaged_image.shape[0] != mask.shape[0] or damaged_image.shape[2:] != mask.shape[2:]:
            raise ValueError("image and mask batch/spatial dimensions must match")

        e1 = self.enc1(torch.cat([damaged_image, mask], dim=1))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self._up_to(b, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up_to(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up_to(d2, e1), e1], dim=1))
        return torch.sigmoid(self.output(d1))
