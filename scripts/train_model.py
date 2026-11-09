from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from orion.models import UNetInpaintingModel
from orion.training_data import SyntheticInpaintingDataset, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORION's compact mask-conditioned U-Net on local clean images.")
    parser.add_argument("--data-dir", required=True, help="Folder containing clean training images (searched recursively)")
    parser.add_argument("--output", default="checkpoints/orion_unet.pt", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=200, help="Number of generated training samples per epoch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-mode", choices=["binary", "soft", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.steps_per_epoch < 1:
        raise SystemExit("epochs, batch-size, and steps-per-epoch must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this PyTorch environment")

    paths = list_images(args.data_dir)
    if not paths:
        raise SystemExit(f"No supported image files found under {args.data_dir}")
    dataset = SyntheticInpaintingDataset(paths, args.image_size, args.steps_per_epoch, args.mask_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = UNetInpaintingModel(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Training on {len(paths)} source images; device={device}; image_size={args.image_size}; mask_mode={args.mask_mode}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for batch in loader:
            clean = batch["clean"].to(device, non_blocking=True)
            damaged = batch["damaged"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            prediction = model(damaged, mask)
            per_pixel = (prediction - clean).abs().mean(dim=1, keepdim=True)
            masked_loss = (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)
            visible_loss = (per_pixel * (1.0 - mask)).mean()
            loss = masked_loss + 0.1 * visible_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())
        print(f"Epoch {epoch:03d}/{args.epochs}: loss={loss_total / max(1, len(loader)):.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "base_channels": args.base_channels,
        "image_size": args.image_size,
        "mask_mode": args.mask_mode,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "seed": args.seed,
    }, output)
    print(f"Saved trained checkpoint: {output.resolve()}")


if __name__ == "__main__":
    main()
