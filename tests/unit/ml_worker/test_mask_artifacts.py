"""Unit tests for the diagnostic mask-rendering helpers.

These PNGs are how the operator visually inspects what the ML models
saw — so the tests pin:

* the exact mask shape is preserved (resize-to-target-size works);
* binary thresholding at 0.5 behaves as documented;
* the overlay actually *changes* the base image inside the mask region
  and leaves it untouched outside (so the operator can trust the
  preview as a diagnostic tool).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from auto_repair_estimator.ml_worker.inference.mask_artifacts import (
    mask_overlay_png_bytes,
    mask_to_grayscale_png_bytes,
)


class TestMaskToGrayscalePng:
    def test_binary_mask_produces_white_pixels_where_mask_is_one(self) -> None:
        # A 4×4 mask with the top-left 2×2 quadrant set.
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[:2, :2] = 1.0

        png = mask_to_grayscale_png_bytes(mask)
        result = np.array(Image.open(io.BytesIO(png)))

        assert result.shape == (4, 4)
        assert result.dtype == np.uint8
        assert (result[:2, :2] == 255).all(), "Mask pixels must render as white."
        assert (result[2:, :] == 0).all(), "Non-mask pixels must render as black."
        assert (result[:, 2:] == 0).all()

    def test_mask_resized_to_target_size_preserves_topology(self) -> None:
        """When target_size is specified, the output PNG has exactly
        those dimensions. This matters because diagnostic PNGs are
        compared against specific crops/frames — mismatched sizes
        would confuse the operator.
        """

        # 2×2 mask, upper row set → upscale to 8×8.
        mask = np.zeros((2, 2), dtype=np.float32)
        mask[0, :] = 1.0

        png = mask_to_grayscale_png_bytes(mask, target_size=(8, 8))
        result = np.array(Image.open(io.BytesIO(png)))

        assert result.shape == (8, 8)
        # Upper half (top 4 rows) should be white, lower half black.
        assert (result[:4, :] == 255).all()
        assert (result[4:, :] == 0).all()

    def test_threshold_at_half_rounds_correctly(self) -> None:
        """The mask helper applies a 0.5 threshold — values exactly at
        0.5 must count as foreground (>=), values below must not."""

        mask = np.array(
            [
                [0.49, 0.50],
                [0.51, 1.00],
            ],
            dtype=np.float32,
        )

        png = mask_to_grayscale_png_bytes(mask)
        result = np.array(Image.open(io.BytesIO(png)))

        assert result[0, 0] == 0, "0.49 < 0.5 must be background."
        assert result[0, 1] == 255, "exactly 0.5 must be foreground (>= threshold)."
        assert result[1, 0] == 255
        assert result[1, 1] == 255

    def test_strips_leading_singleton_dim_from_yolo_output(self) -> None:
        """YOLO masks often come with a leading (1, H, W) shape; the
        helper must handle that without an error."""

        mask = np.zeros((1, 2, 2), dtype=np.float32)
        mask[0, 0, 0] = 1.0

        png = mask_to_grayscale_png_bytes(mask)
        result = np.array(Image.open(io.BytesIO(png)))
        assert result.shape == (2, 2)
        assert result[0, 0] == 255
        assert result[1, 1] == 0


class TestMaskOverlayPng:
    def test_overlay_paints_masked_region_and_leaves_rest_untouched(self) -> None:
        """The overlay's diagnostic value depends on this: unmasked
        pixels must be identical to the base image so the operator
        can trust the overlay as a ground-truth visual reference."""

        base_rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
        base = Image.fromarray(base_rgb)

        mask = np.zeros((4, 4), dtype=np.float32)
        mask[0, 0] = 1.0  # single pixel

        png = mask_overlay_png_bytes(base, mask, color=(255, 0, 0), alpha=0.5)
        rendered = np.array(Image.open(io.BytesIO(png)))

        # Unmasked pixels — must match base exactly.
        assert (rendered[1:, :] == 100).all()
        assert (rendered[:, 1:] == 100).all()
        # Masked pixel — blended: 0.5*100 + 0.5*[255,0,0] = [177, 50, 50].
        assert rendered[0, 0, 0] == 177
        assert rendered[0, 0, 1] == 50
        assert rendered[0, 0, 2] == 50

    def test_rejects_invalid_alpha(self) -> None:
        base = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
        mask = np.ones((2, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="alpha"):
            mask_overlay_png_bytes(base, mask, color=(0, 0, 0), alpha=1.5)

    def test_mask_resized_to_base_dimensions_if_smaller(self) -> None:
        """Masks from YOLO often come at 160×160 regardless of input
        size; the overlay must resize them to the base image before
        blending, otherwise the call would raise on shape mismatch."""

        base_rgb = np.full((8, 8, 3), 50, dtype=np.uint8)
        base = Image.fromarray(base_rgb)
        # 2×2 mask: upper-left quadrant set → after NN upscale to 8×8
        # covers the upper-left 4×4.
        mask = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

        png = mask_overlay_png_bytes(base, mask, color=(200, 200, 200), alpha=0.5)
        rendered = np.array(Image.open(io.BytesIO(png)))

        # Upper-left 4×4 got blended.
        assert rendered[0, 0, 0] == int(0.5 * 50 + 0.5 * 200)
        # Bottom-right stays untouched.
        assert rendered[7, 7, 0] == 50
