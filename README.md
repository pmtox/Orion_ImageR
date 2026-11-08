# ORION — Occlusion Reconstruction and Image Optimization Network

ORION is a focused image-inpainting project. Its research question is whether a reconstruction system can use image context and damage-mask information to fill missing content, and under what conditions soft masks help compared with binary masks.

## Current status

- **IMPLEMENTED:** image preprocessing, binary/soft mask generation, synthetic corruption, OpenCV Telea inpainting baseline, a mask-conditioned CNN baseline architecture, image quality metrics, a Streamlit demo, and unit tests.
- **NOT YET IMPLEMENTED:** training/data pipeline for the CNN, a trained checkpoint, controlled binary-vs-soft experiments, and experimental results.
- **NOT VERIFIED HERE:** dependency installation, GPU use, application launch, and test execution. Run the checks below in your environment and record their actual outcomes.

The app currently uses OpenCV Telea as its dependable reconstruction method. The CNN is provided as a model baseline, but is not presented as trained or as producing meaningful learned reconstructions until a training pipeline and checkpoint exist. OpenCV's classical method requires a binary mask, so soft masks are thresholded for that baseline; the CNN interface accepts the soft-valued mask directly.

## Run locally

Python 3.10+ is recommended. Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py