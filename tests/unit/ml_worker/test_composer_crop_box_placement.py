"""Tests for the crop-box-aware mask placement added to ``compose``.

Before this change, a damage mask produced by the detector on a *crop*
was naively resized to fill the whole original image, producing the
"smear across the entire car" artefact visible in the user's screenshot.
The fix: detections now carry ``crop_box_pixels`` (the absolute pixel
rectangle of their source crop on the original image), and the composer
blits each crop-local mask back into exactly that rectangle.

These tests pin that contract:

* Pixels **inside** the crop rectangle change colour.
* Pixels **outside** stay unchanged — critical, because this is what the
  old "resize-to-full-frame" bug violated.
* Degenerate boxes (zero area, fully out of frame) are skipped without
  raising, so one bad detection doesn't kill the whole compose step.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from auto_repair_estimator.ml_worker.inference.composer import compose


@dataclass
class _Detection:
    damage_type: str
    mask: Any | None
    crop_box_pixels: tuple[int, int, int, int] | None = None


def _grey(w: int, h: int, value: int = 100) -> Image.Image:
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _decoded(jpeg_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"), dtype=np.float32)


class TestMaskPlacementByCropBox:
    def test_mask_stays_inside_crop_box_only(self) -> None:
        """A crop covering the RIGHT half of the image should leave the
        LEFT half untouched. The old full-frame resize bug would have
        changed every pixel — this test detects that regression.
        """

        img_w, img_h = 400, 300
        img = _grey(img_w, img_h, value=100)
        crop_w, crop_h = 200, 300  # right half in pixels
        mask = np.ones((crop_h, crop_w), dtype=np.float32)
        crop_box = (200, 0, 400, 300)  # x1, y1, x2, y2

        out = compose(
            img,
            [_Detection(damage_type="scratch", mask=mask, crop_box_pixels=crop_box)],
        )
        decoded = _decoded(out)

        # Left half should be ~unchanged (JPEG quantisation noise only).
        left = decoded[:, : img_w // 2, :]
        diff_left = float(np.abs(left - 100).mean())
        assert diff_left < 5.0, (
            f"Left half was modified (diff_left={diff_left:.2f}). "
            "The mask must stay inside crop_box_pixels only."
        )

        # Right half should be visibly red (scratch = (255, 0, 0)).
        right = decoded[:, img_w // 2 :, :]
        assert right[:, :, 0].mean() > 130, "Right-half R channel must rise after red scratch overlay."

    def test_mask_at_specific_sub_region_only_paints_there(self) -> None:
        """A 50x50 mask at offset (100, 100) must paint only that square."""

        img_w, img_h = 300, 300
        img = _grey(img_w, img_h, value=50)
        mask = np.ones((50, 50), dtype=np.float32)
        crop_box = (100, 100, 150, 150)

        out = compose(
            img,
            [_Detection(damage_type="dent", mask=mask, crop_box_pixels=crop_box)],
        )
        decoded = _decoded(out)

        # Inside the crop rectangle: dent = GREEN (0, 255, 0) — G channel should spike.
        inside_g = decoded[100:150, 100:150, 1].mean()
        assert inside_g > 100, f"G channel inside crop should be high (got {inside_g:.1f})."

        # Outside: a corner far from the rectangle — must remain close to 50.
        corner = decoded[0:50, 0:50, :]
        diff_corner = float(np.abs(corner - 50).mean())
        assert diff_corner < 5.0, (
            f"Corner far from mask region was modified (diff={diff_corner:.2f})."
        )

    def test_degenerate_crop_box_is_skipped_silently(self) -> None:
        """An inverted/zero-area ``crop_box_pixels`` must not raise and
        must not paint anything. Legitimate detections in the same
        compose call should still render — one bad detection cannot
        poison the whole composite.
        """

        img_w, img_h = 200, 200
        img = _grey(img_w, img_h, value=80)
        bad = _Detection(
            damage_type="scratch",
            mask=np.ones((50, 50), dtype=np.float32),
            crop_box_pixels=(150, 150, 100, 100),  # x2<x1, y2<y1
        )
        good = _Detection(
            damage_type="rust",
            mask=np.ones((40, 40), dtype=np.float32),
            crop_box_pixels=(10, 10, 50, 50),
        )

        out = compose(img, [bad, good])
        decoded = _decoded(out)

        # Good detection's area should show rust (255, 165, 0) — R channel up.
        good_r = decoded[10:50, 10:50, 0].mean()
        assert good_r > 120, "Good detection must still render after a bad sibling."

    def test_legacy_detection_without_crop_box_uses_full_image_fallback(self) -> None:
        """Back-compat: ``crop_box_pixels=None`` (e.g. older test doubles)
        falls through to the old "resize mask to full image" behaviour
        so the existing composer tests keep working.
        """

        img_w, img_h = 120, 120
        img = _grey(img_w, img_h, value=30)
        mask = np.ones((120, 120), dtype=np.float32)  # already image-size

        out = compose(
            img,
            [_Detection(damage_type="scratch", mask=mask, crop_box_pixels=None)],
        )
        decoded = _decoded(out)
        # The whole frame should shift toward red.
        assert decoded[:, :, 0].mean() > 80
