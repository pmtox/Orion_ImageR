# ORION — Occlusion Reconstruction and Image Optimization Network

ORION is an image-inpainting application: upload an image, create a synthetic damaged region, and reconstruct it using either OpenCV Telea or a locally trained, mask-conditioned U-Net.

## Current implementation

- **IMPLEMENTED:** Streamlit upload/demo, binary and soft rectangular masks, synthetic corruption, OpenCV Telea baseline, compact mask-conditioned U-Net, image metrics, generated-on-demand training data, training script, checkpoint inference, regression tests.
- **TRAINED MODEL:** Not included in Git. Train your own checkpoint from a folder of clean images using the steps below.
- **VERIFICATION:** The project maintainer reported 8 tests passing before the U-Net training path was added. Run the updated tests locally; this session cannot execute the project or verify GPU behavior.

The CNN in `src/orion/models/cnn.py` remains a small baseline architecture. The application uses the U-Net when `checkpoints/orion_unet.pt` exists; otherwise Telea is available.

## Quick start (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pytest
py -m streamlit run app.py
```

If you already have the environment and tests working, update the repo and run `py -3.13 -m pytest` again.

## Train the U-Net

Put clean training images in a folder (subfolders are searched recursively), then run from the project root:

```powershell
py -3.13 scripts/train_model.py --data-dir "D:\path\to\clean_images" --epochs 10 --steps-per-epoch 200 --batch-size 4 --image-size 128 --device auto
```

The script creates `checkpoints/orion_unet.pt`. It generates random rectangular masks and solid-color corruption during training. `--mask-mode` accepts `binary`, `soft`, or `mixed` (default). Use a legitimate image collection for which you have permission to train; do not put large datasets or checkpoints into Git.

For a quick smoke test before a longer run:

```powershell
py -3.13 scripts/train_model.py --data-dir "D:\path\to\clean_images" --epochs 1 --steps-per-epoch 8 --batch-size 2 --image-size 64
```

This verifies the training path, not output quality. Training on a small or narrow image collection may produce poor generalization. Start modestly on the GTX 1650 and lower batch size or resolution if CUDA runs out of memory.

After training, restart/reload the Streamlit app if needed. The **Trained U-Net** method will appear when the checkpoint is present. If no model is trained yet, use **OpenCV Telea**.

## Data and mask behavior

- Mask `0` means visible; `1` means damaged; intermediate values are soft damage strength.
- The U-Net receives normalized damaged RGB and the mask as a fourth input channel.
- The output is composited with the input so pixels outside the mask are preserved.
- Telea converts any positive mask value to an inpainting region, so it does not use graded soft-mask values.
- Metrics are against the clean uploaded image because the app synthesizes the damage from that image.

Training and output files are local and ignored by Git. Never describe a checkpoint or model quality as verified until you have trained and tested it locally.
