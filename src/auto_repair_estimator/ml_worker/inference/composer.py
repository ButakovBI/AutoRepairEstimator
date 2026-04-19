from __future__ import annotations

import io

import numpy as np
from loguru import logger
from PIL import Image

ALPHA = 0.4
MASK_THRESHOLD = 0.5

DAMAGE_COLORS: dict[str, tuple[int, int, int]] = {
    "scratch": (255, 0, 0),
    "dent": (0, 255, 0),
    "crack": (0, 0, 255),
    "rust": (255, 165, 0),
    "paint_chip": (255, 255, 0),
    "broken_glass": (0, 255, 255),
    "flat_tire": (128, 0, 128),
    "broken_headlight": (255, 105, 180),
}
DEFAULT_COLOR = (255, 0, 255)


def compose(original_image: Image.Image, damage_detections: list) -> bytes:  # type: ignore[type-arg]
    """
    Overlays damage masks onto the original image using alpha blending.
    X^comp = (1 - alpha * M) * X + alpha * M * C
    """
    img_arr = np.array(original_image.convert("RGB"), dtype=np.float32)
    h, w = img_arr.shape[:2]
    composite = img_arr.copy()

    masks_applied = 0
    for detection in damage_detections:
        if detection.mask is None:
            continue

        mask = np.array(detection.mask, dtype=np.float32)
        if mask.shape != (h, w):
            from PIL import Image as PImage

            mask_img = PImage.fromarray((mask * 255).astype(np.uint8)).resize((w, h), PImage.NEAREST)
            mask = np.array(mask_img, dtype=np.float32) / 255.0

        binary_mask = (mask >= MASK_THRESHOLD).astype(np.float32)

        color = DAMAGE_COLORS.get(detection.damage_type, DEFAULT_COLOR)
        color_arr = np.array(color, dtype=np.float32)

        for c in range(3):
            composite[:, :, c] = (1.0 - ALPHA * binary_mask) * composite[:, :, c] + ALPHA * binary_mask * color_arr[c]
        masks_applied += 1

    composite = np.clip(composite, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(composite)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG", quality=90)
    logger.debug("Composed {} masks onto image", masks_applied)
    return buf.getvalue()
