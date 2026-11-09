from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .corruption import apply_corruption
from .masks import make_rect_mask
from .preprocessing import normalize_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: str | Path) -> list[Path]:
    """Find supported image files recursively beneath root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


class SyntheticInpaintingDataset(Dataset):
    """Generate image/mask/corruption training samples from clean images."""

    def __init__(self, image_paths: list[Path], image_size: int = 128, samples_per_epoch: int = 500, mask_mode: str = "mixed"):
        if not image_paths:
            raise ValueError("No supported images found in the training directory")
        if image_size < 32 or samples_per_epoch <= 0:
            raise ValueError("image_size must be >=32 and samples_per_epoch must be positive")
        if mask_mode not in {"binary", "soft", "mixed"}:
            raise ValueError("mask_mode must be binary, soft, or mixed")
        self.paths = image_paths
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        path = random.choice(self.paths)
        with Image.open(path) as image_file:
            image = ImageOps.fit(image_file.convert("RGB"), (self.image_size, self.image_size), method=Image.Resampling.BICUBIC)
        clean = np.asarray(image, dtype=np.uint8).copy()
        h, w = clean.shape[:2]
        rect_w = random.randint(max(8, w // 8), max(9, w * 3 // 5))
        rect_h = random.randint(max(8, h // 8), max(9, h * 3 // 5))
        x = random.randint(0, max(0, w - rect_w))
        y = random.randint(0, max(0, h - rect_h))
        soft = self.mask_mode == "soft" or (self.mask_mode == "mixed" and random.random() < 0.5)
        edge_width = random.randint(4, max(5, self.image_size // 10)) if soft else 0
        mask = make_rect_mask(h, w, x=x, y=y, rect_width=rect_w, rect_height=rect_h, soft=soft, edge_width=edge_width)
        damaged = apply_corruption(clean, mask, seed=random.randrange(0, 2**31))

        clean_chw = np.transpose(normalize_image(clean), (2, 0, 1)).copy()
        damaged_chw = np.transpose(normalize_image(damaged), (2, 0, 1)).copy()
        mask_chw = mask[None, ...].astype(np.float32, copy=True)
        return {
            "clean": torch.from_numpy(clean_chw),
            "damaged": torch.from_numpy(damaged_chw),
            "mask": torch.from_numpy(mask_chw),
        }
