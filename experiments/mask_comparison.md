# ORION mask comparison experiment

## Purpose

Run a reproducible **classical-baseline** comparison with the same source image, rectangular mask coordinates, corruption seed, and Telea radius for binary and soft masks. This produces outputs and metrics; it does not train a model and does not establish whether soft-mask conditioning helps a neural network.

## Run on Windows PowerShell

From the repository root:

```powershell
py -3.13 scripts/compare_masks.py --image "data/sample_images/example.jpg" --output-dir "outputs/mask_comparison"
```

Optional controls include `--x`, `--y`, `--mask-width`, `--mask-height`, `--edge-width`, `--seed`, `--radius`, and `--max-side`. Both runs use the same rectangle and corruption color seed. To use the original image resolution, pass `--max-side 0`.

The script writes binary/soft masks, corrupted images, Telea outputs, `metrics.csv`, and `comparison.json` into the output directory. Review the report for the exact input path and settings before recording the run in `experiment_log.md`.

## Interpretation limitation

`src/orion/classical.py` converts any positive mask value to 255 before calling OpenCV Telea. The current binary and soft rectangular masks have the same positive spatial support, so Telea receives the same inpainting region in both cases. Since it replaces that region regardless of the corruption color values, the two reconstructed images and metrics may be identical. This is expected for this baseline and is **not evidence that soft masks do or do not help**. The central research comparison needs a trained mask-conditioned model that consumes the graded mask values.
