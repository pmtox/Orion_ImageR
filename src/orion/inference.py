from __future__ import annotations

from pathlib import Path

import torch

from .models import UNetInpaintingModel


def load_unet_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> UNetInpaintingModel:
    """Load a checkpoint produced by scripts/train_model.py."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Older supported PyTorch releases do not accept weights_only.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict")
    model = UNetInpaintingModel(base_channels=int(checkpoint.get("base_channels", 16)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def reconstruct_with_unet(image_rgb, mask, model: UNetInpaintingModel, device: str | torch.device = "cpu"):
    """Reconstruct masked pixels and preserve unmasked source pixels."""
    import numpy as np

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError("image_rgb must be HxWx3 RGB uint8")
    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("mask shape must match image")
    if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("mask values must be finite and in [0,1]")
    image_tensor = torch.from_numpy(np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1)).copy()).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32, copy=True)).unsqueeze(0).unsqueeze(0).to(device)
    model = model.to(device).eval()
    with torch.inference_mode():
        prediction = model(image_tensor, mask_tensor)[0].cpu().numpy().transpose(1, 2, 0)
    alpha = mask.astype(np.float32)[..., None]
    result = image_rgb.astype(np.float32) * (1.0 - alpha) + prediction * 255.0 * alpha
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
