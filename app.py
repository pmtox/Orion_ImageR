from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from orion.classical import inpaint_telea  # noqa: E402
from orion.corruption import apply_corruption  # noqa: E402
from orion.masks import make_rect_mask  # noqa: E402
from orion.metrics import image_metrics  # noqa: E402
from orion.preprocessing import load_rgb  # noqa: E402

st.set_page_config(page_title="ORION Inpainting", page_icon="🖼️", layout="wide")
st.title("ORION — Image Inpainting")
st.caption("Synthetic damage → mask-aware corruption → classical reconstruction baseline")
st.info(
    "This first demo uses OpenCV Telea inpainting. The CNN is not trained yet; "
    "no learned reconstruction results are claimed."
)

upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
if upload is None:
    st.markdown("Upload an image to create a synthetic rectangular damaged region and compare outputs.")
    st.stop()

try:
    image = load_rgb(upload, max_side=512)
except Exception as exc:
    st.error(f"Could not load image: {exc}")
    st.stop()

height, width = image.shape[:2]
st.write(f"Image: **{width} × {height}** pixels (resized only if its longest side exceeds 512).")

with st.form("inpainting_controls"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        x = st.slider("Mask left", 0, max(0, width - 1), width // 4)
        mask_width = st.slider("Mask width", 1, width, max(1, width // 3))
    with col_b:
        y = st.slider("Mask top", 0, max(0, height - 1), height // 4)
        mask_height = st.slider("Mask height", 1, height, max(1, height // 3))
    with col_c:
        mask_kind = st.selectbox("Mask", ["Binary", "Soft"])
        soft_edge = st.slider("Soft edge width (pixels)", 1, 40, 12, disabled=mask_kind == "Binary")
    seed = st.number_input("Corruption seed", min_value=0, max_value=2**31 - 1, value=42)
    run = st.form_submit_button("Generate corruption and inpaint", type="primary")

if run:
    mask = make_rect_mask(
        height, width, x=x, y=y, rect_width=mask_width, rect_height=mask_height,
        soft=(mask_kind == "Soft"), edge_width=soft_edge if mask_kind == "Soft" else 0,
    )
    corrupted = apply_corruption(image, mask, seed=int(seed))
    reconstruction = inpaint_telea(corrupted, mask)
    metrics = image_metrics(image, reconstruction, mask)

    st.session_state["orion_result"] = {
        "image": image,
        "mask": mask,
        "corrupted": corrupted,
        "reconstruction": reconstruction,
        "metrics": metrics,
    }

result = st.session_state.get("orion_result")
if result:
    st.subheader("Results")
    cols = st.columns(4)
    panels = [
        ("Original", result["image"]),
        ("Damage mask (white = damaged)", (result["mask"] * 255).astype(np.uint8)),
        ("Corrupted", result["corrupted"]),
        ("Telea reconstruction", result["reconstruction"]),
    ]
    for col, (label, array) in zip(cols, panels):
        with col:
            st.image(array, caption=label, use_container_width=True, clamp=True)

    st.subheader("Synthetic-reference metrics")
    st.json(result["metrics"])
    st.caption("Metrics compare against the original because this demo generated the damage synthetically.")

    for label, key in [("Download reconstruction", "reconstruction"), ("Download corrupted image", "corrupted")]:
        buf = BytesIO()
        Image.fromarray(result[key]).save(buf, format="PNG")
        st.download_button(label, data=buf.getvalue(), file_name=f"orion_{key}.png", mime="image/png")
