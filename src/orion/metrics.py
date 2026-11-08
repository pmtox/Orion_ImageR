from __future__ import annotations

import numpy as np


def image_metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute full-image MAE/PSNR, masked MAE, and global SSIM (RGB, [0,255])."""
    if reference.shape != prediction.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("reference and prediction must be matching HxWx3 images")
    if mask.shape != reference.shape[:2]:
        raise ValueError("mask spatial shape must match images")
    ref = reference.astype(np.float64) / 255.0
    pred = prediction.astype(np.float64) / 255.0
    abs_error = np.abs(ref - pred)
    mae = float(abs_error.mean())
    damaged = mask > 0
    masked_mae = float(abs_error[damaged].mean()) if np.any(damaged) else 0.0
    mse = float(np.mean((ref - pred) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))

    # Global SSIM averaged over channels, following the standard luminance/contrast form.
    c1, c2 = 0.01**2, 0.03**2
    ssim_channels = []
    for channel in range(3):
        a, b = ref[..., channel], pred[..., channel]
        mu_a, mu_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var()), float(b.var())
        cov = float(np.mean((a - mu_a) * (b - mu_b)))
        ssim_channels.append(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))
    return {"mae": mae, "masked_mae": masked_mae, "psnr_db": psnr, "ssim_global": float(np.mean(ssim_channels))}
