import numpy as np
import pytest
import torch
from PIL import Image

from orion.corruption import apply_corruption
from orion.inference import reconstruct_with_unet
from orion.masks import make_rect_mask
from orion.metrics import image_metrics
from orion.models import CNNInpaintingBaseline, UNetInpaintingModel
from orion.training_data import SyntheticInpaintingDataset, list_images


def test_binary_mask_values_and_extent():
    mask = make_rect_mask(8, 10, x=2, y=1, rect_width=4, rect_height=3)
    assert mask.shape == (8, 10)
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask.sum() == 12


def test_binary_mask_allows_zero_edge_width():
    mask = make_rect_mask(8, 10, x=2, y=1, rect_width=4, rect_height=3, edge_width=0)
    assert mask.sum() == 12
    assert set(np.unique(mask)) == {0.0, 1.0}


def test_soft_mask_has_graded_boundary():
    mask = make_rect_mask(12, 12, x=2, y=2, rect_width=8, rect_height=8, soft=True, edge_width=3)
    assert mask.min() == 0
    assert mask.max() == 1
    assert np.any((mask > 0) & (mask < 1))


def test_soft_mask_rejects_nonpositive_edge_width():
    with pytest.raises(ValueError, match="edge_width must be positive"):
        make_rect_mask(8, 8, x=1, y=1, rect_width=4, rect_height=4, soft=True, edge_width=0)


def test_corruption_preserves_visible_pixels_and_is_seeded():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    mask = make_rect_mask(8, 8, x=2, y=2, rect_width=3, rect_height=3)
    a = apply_corruption(image, mask, seed=7)
    b = apply_corruption(image, mask, seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[mask == 0], image[mask == 0])
    assert np.any(a[mask == 1] != image[mask == 1])


def test_empty_mask_is_valid_for_corruption_and_metrics():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.float32)
    np.testing.assert_array_equal(apply_corruption(image, mask), image)
    result = image_metrics(image, image, mask)
    assert result["masked_mae"] == 0
    assert result["mae"] == 0
    assert result["psnr_db"] == float("inf")


def test_cnn_accepts_four_channel_conditioning_and_returns_rgb():
    model = CNNInpaintingBaseline(width=8)
    image = torch.rand(2, 3, 16, 16)
    mask = torch.rand(2, 1, 16, 16)
    result = model(image, mask)
    assert result.shape == image.shape
    assert torch.all((result >= 0) & (result <= 1))


def test_unet_accepts_non_divisible_shapes():
    model = UNetInpaintingModel(base_channels=4)
    image = torch.rand(1, 3, 65, 73)
    mask = torch.rand(1, 1, 65, 73)
    result = model(image, mask)
    assert result.shape == image.shape
    assert torch.all((result >= 0) & (result <= 1))


def test_unet_inference_preserves_unmasked_pixels():
    model = UNetInpaintingModel(base_channels=4)
    image = np.full((32, 32, 3), 80, dtype=np.uint8)
    mask = make_rect_mask(32, 32, x=8, y=8, rect_width=12, rect_height=12)
    damaged = apply_corruption(image, mask, seed=4)
    output = reconstruct_with_unet(damaged, mask, model)
    np.testing.assert_array_equal(output[mask == 0], image[mask == 0])


def test_training_dataset_produces_model_ready_tensors(tmp_path):
    image = Image.fromarray(np.full((48, 64, 3), (40, 100, 180), dtype=np.uint8))
    image.save(tmp_path / "sample.png")
    paths = list_images(tmp_path)
    dataset = SyntheticInpaintingDataset(paths, image_size=32, samples_per_epoch=2, mask_mode="mixed")
    item = dataset[0]
    assert item["clean"].shape == (3, 32, 32)
    assert item["damaged"].shape == (3, 32, 32)
    assert item["mask"].shape == (1, 32, 32)
    assert torch.isfinite(item["mask"]).all()
    assert torch.all((item["mask"] >= 0) & (item["mask"] <= 1))
