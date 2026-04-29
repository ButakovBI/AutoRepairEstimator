"""Adversarial tests for ``composer.compose`` targeting branches and inputs
that the main ``test_composer.py`` suite does not exercise.

Targeted gaps:

1. **Mask shape mismatch** — in production the mask comes from the YOLO
   detector at the model's inference resolution (e.g. 640x640) while the
   original image can be any size. ``compose`` contains a branch that
   resizes the mask; this branch is currently **untested**, so a regression
   to the resize logic would silently compose a black square onto the
   image.
2. **Unknown damage_type** — the ``DAMAGE_COLORS`` dict has only 8 keys.
   An ML upgrade that emits a new damage class must not crash the composer.
3. **Empty damage list does not re-encode visible differences** — regression
   guard to ensure JPEG round-trip alone doesn't hide the "no-op" path.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from auto_repair_estimator.ml_worker.inference.composer import DAMAGE_COLORS, DEFAULT_COLOR, compose


@dataclass
class _FakeDamage:
    damage_type: str
    mask: Any | None


def _image(width: int, height: int, rgb: tuple[int, int, int] = (50, 60, 70)) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = rgb
    return Image.fromarray(arr)


def test_mask_smaller_than_image_is_resized_and_applied() -> None:
    """The mask is 200x200 and covers the full frame; the image is 640x480.

    The resize branch must upscale the mask to (width=640, height=480)
    using NEAREST, then apply the red scratch colour everywhere. Every
    pixel in the output therefore has significant non-original content.
    """

    img_w, img_h = 640, 480
    img = _image(img_w, img_h, rgb=(50, 60, 70))
    # 200x200 full-white mask — mask.shape is (200, 200), image is (480, 640)
    small_mask = np.ones((200, 200), dtype=np.float32)

    result = compose(img, [_FakeDamage("scratch", small_mask)])

    decoded = np.array(Image.open(io.BytesIO(result)).convert("RGB"), dtype=np.float32)
    # ``scratch`` is RED (255, 0, 0). Alpha blending with alpha=0.4 over
    # original (50, 60, 70) yields approximately ((1-0.4)*50+0.4*255, ...)
    # = (132, 36, 42). The R channel should have dramatically increased.
    mean_r = float(decoded[:, :, 0].mean())
    mean_g = float(decoded[:, :, 1].mean())

    assert mean_r > 100, (
        f"Red channel mean {mean_r:.1f} — the full-frame resized mask must "
        "raise it well above the original 50."
    )
    assert mean_g < 80, (
        f"Green channel mean {mean_g:.1f} — scratch colour has zero green; "
        "alpha-blended output should reduce G slightly, not increase it."
    )


def test_mask_larger_than_image_is_also_resized_down() -> None:
    """Symmetric case: mask is bigger than the image.

    Even though YOLO typically pads/letterboxes to model input size, a
    non-letterboxed mask might arrive from a newer model. The resize
    branch covers both upscale and downscale.
    """

    img_w, img_h = 100, 100
    img = _image(img_w, img_h)
    # 640x640 mask covers whole frame.
    big_mask = np.ones((640, 640), dtype=np.float32)

    result = compose(img, [_FakeDamage("dent", big_mask)])

    decoded = np.array(Image.open(io.BytesIO(result)).convert("RGB"), dtype=np.float32)
    # ``dent`` is GREEN (0, 255, 0). Green channel must rise markedly.
    assert decoded[:, :, 1].mean() > 100


def test_unknown_damage_type_uses_default_color_without_crashing() -> None:
    """A future damage class that isn't in ``DAMAGE_COLORS`` must not
    raise. The composer falls back to ``DEFAULT_COLOR`` (magenta).
    """

    # Guard against the defaults being reshuffled — this test has meaning
    # only while ``DEFAULT_COLOR`` is distinct from all registered colours.
    assert DEFAULT_COLOR not in DAMAGE_COLORS.values()

    img = _image(320, 320, rgb=(0, 0, 0))
    mask = np.ones((320, 320), dtype=np.float32)

    result = compose(img, [_FakeDamage("brand_new_damage_kind", mask)])

    decoded = np.array(Image.open(io.BytesIO(result)).convert("RGB"), dtype=np.float32)
    # Magenta (255, 0, 255): R and B should spike, G stays low.
    assert decoded[:, :, 0].mean() > 50  # R
    assert decoded[:, :, 2].mean() > 50  # B
    assert decoded[:, :, 1].mean() < 30  # G


def test_mask_below_threshold_does_not_paint_any_pixels() -> None:
    """The mask threshold (MASK_THRESHOLD = 0.5) must gate every pixel.

    A mask of uniform 0.4 (all values below the threshold) must leave the
    image visually unchanged after JPEG round-trip.
    """

    img_w, img_h = 320, 320
    img = _image(img_w, img_h, rgb=(100, 100, 100))
    # Same shape as image, all below threshold.
    below_threshold_mask = np.full((img_h, img_w), 0.4, dtype=np.float32)
    orig = np.array(img, dtype=np.float32)

    result = compose(img, [_FakeDamage("rust", below_threshold_mask)])

    decoded = np.array(Image.open(io.BytesIO(result)).convert("RGB"), dtype=np.float32)
    # JPEG quantisation adds some noise; mean absolute diff under 3 is
    # what "no mask applied" looks like vs >20 when a real mask paints.
    diff = float(np.abs(decoded - orig).mean())
    assert diff < 5.0, f"Sub-threshold mask should not paint pixels (diff={diff:.2f})."
