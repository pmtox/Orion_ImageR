from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orion.classical import inpaint_telea  # noqa: E402
from orion.corruption import apply_corruption  # noqa: E402
from orion.masks import make_rect_mask  # noqa: E402
from orion.metrics import image_metrics  # noqa: E402
from orion.preprocessing import load_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled binary-vs-soft mask comparison using the Telea baseline."
    )
    parser.add_argument("--image", required=True, help="Path to a clean reference image")
    parser.add_argument("--output-dir", default="outputs/mask_comparison", help="Directory for images and reports")
    parser.add_argument("--max-side", type=int, default=512, help="Downscale longest image side to this size (0 disables)")
    parser.add_argument("--x", type=int, default=None, help="Rectangle left coordinate")
    parser.add_argument("--y", type=int, default=None, help="Rectangle top coordinate")
    parser.add_argument("--mask-width", type=int, default=None, help="Rectangle width")
    parser.add_argument("--mask-height", type=int, default=None, help="Rectangle height")
    parser.add_argument("--edge-width", type=int, default=12, help="Soft-mask boundary ramp width in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic corruption color")
    parser.add_argument("--radius", type=float, default=3.0, help="OpenCV Telea inpainting radius")
    return parser.parse_args()


def json_metric(value: float) -> float | str:
    """Keep JSON valid if a perfect reconstruction yields infinite PSNR."""
    return float(value) if np.isfinite(value) else str(value)


def main() -> None:
    args = parse_args()
    if args.max_side < 0:
        raise SystemExit("--max-side must be zero (no resize) or a positive integer")
    if args.edge_width <= 0:
        raise SystemExit("--edge-width must be positive")

    image = load_rgb(args.image, max_side=args.max_side or None)
    height, width = image.shape[:2]
    x = args.x if args.x is not None else width // 4
    y = args.y if args.y is not None else height // 4
    rect_width = args.mask_width if args.mask_width is not None else max(1, width // 3)
    rect_height = args.mask_height if args.mask_height is not None else max(1, height // 3)
    if x < 0 or y < 0 or x >= width or y >= height:
        raise SystemExit(f"Mask origin ({x}, {y}) must lie inside image dimensions {width}x{height}")
    if rect_width <= 0 or rect_height <= 0:
        raise SystemExit("Mask width and height must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for mask_type, soft in (("binary", False), ("soft", True)):
        mask = make_rect_mask(
            height,
            width,
            x=x,
            y=y,
            rect_width=rect_width,
            rect_height=rect_height,
            soft=soft,
            edge_width=args.edge_width,
        )
        damaged = apply_corruption(image, mask, seed=args.seed)
        reconstruction = inpaint_telea(damaged, mask, radius=args.radius)
        metrics = image_metrics(image, reconstruction, mask)

        Image.fromarray((mask * 255).round().astype(np.uint8)).save(output_dir / f"{mask_type}_mask.png")
        Image.fromarray(damaged).save(output_dir / f"{mask_type}_corrupted.png")
        Image.fromarray(reconstruction).save(output_dir / f"{mask_type}_telea.png")
        results[mask_type] = {
            "mask_positive_pixels": int(np.count_nonzero(mask > 0)),
            "mask_sum": float(mask.sum()),
            "metrics": {key: json_metric(value) for key, value in metrics.items()},
        }

    # Telea thresholds masks at mask > 0. Keep this explicit: this run tests
    # classical-baseline behavior, not whether a learned model uses soft weights.
    binary = make_rect_mask(height, width, x=x, y=y, rect_width=rect_width, rect_height=rect_height)
    soft_mask = make_rect_mask(
        height, width, x=x, y=y, rect_width=rect_width, rect_height=rect_height,
        soft=True, edge_width=args.edge_width,
    )
    same_effective_region = bool(np.array_equal(binary > 0, soft_mask > 0))
    comparison = {
        "status": "EXPERIMENTAL_RUNNER; results are created only when run locally",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image": str(Path(args.image).resolve()),
        "image_dimensions": {"width": width, "height": height},
        "mask_convention": "0=visible, 1=damaged; intermediate soft values are graded corruption strength",
        "settings": {
            "x": x, "y": y, "mask_width": rect_width, "mask_height": rect_height,
            "soft_edge_width": args.edge_width, "corruption_seed": args.seed,
            "telea_radius": args.radius, "max_side": args.max_side,
        },
        "telea_limitation": (
            "The current Telea adapter converts every positive mask value to a binary inpainting region. "
            "If binary and soft masks have the same positive support, Telea does not consume their graded values; "
            "this is a classical baseline comparison, not evidence about learned soft-mask benefits."
        ),
        "same_positive_mask_region": same_effective_region,
        "runs": results,
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["mask_type", "mask_positive_pixels", "mask_sum", "mae", "masked_mae", "psnr_db", "ssim_global"])
        for mask_type, record in results.items():
            metrics = record["metrics"]
            writer.writerow([
                mask_type, record["mask_positive_pixels"], record["mask_sum"],
                metrics["mae"], metrics["masked_mae"], metrics["psnr_db"], metrics["ssim_global"],
            ])

    print(f"Saved controlled comparison to: {output_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Same positive mask region: {same_effective_region}")
    print("Note: Telea binarizes the mask; use a trained mask-conditioned model to test graded-mask conditioning.")
    for mask_type, record in results.items():
        print(f"{mask_type}: {record['metrics']}")


if __name__ == "__main__":
    main()
