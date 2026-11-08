import numpy as np
import pytest
import torch

from orion.corruption import apply_corruption
from orion.masks import make_rect_mask
from orion.metrics import image_metrics
from orion.models import CNNInpaintingBaseline


def test_binary_mask_values_and_extent():
    mask = make_rect_mask(8, 10, x=2, y=1, rect_width=4, rect_height=3)
    assert mask.shape == (8, 10)
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask.sum() == 12


def test_soft_mask_has_graded_boundary():
    mask = make_rect_mask(12, 12, x=2, y=2, rect_width=8, rect_height=8, soft=True, edge_width=3)
    assert mask.min() == 0
    assert mask.max() == 1
    assert np.any((mask > 0) & (mask < 1))


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


def test_corruption_rejects_mismatched_mask():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_corruption(image, np.zeros((3, 4), dtype=np.float32))
