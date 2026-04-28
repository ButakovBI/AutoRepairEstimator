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
    """Overlay damage masks onto ``original_image`` with alpha blending.

    Why the crop-box indirection matters: the damage detector runs on each
    part *crop*, so ``detection.mask`` is in crop-local pixel coordinates.
    Previously the composer just resized every mask to the full image
    size, which smeared a door-damage mask across the entire car body (the
    regression the user reported via screenshot). Now we honour
    ``detection.crop_box_pixels`` and blit each mask back at its source
    crop's location on the original frame. Legacy callers that pass
    original-image-space masks (no crop box) still work via the fallback
    branch below.

    Blend formula per channel:
        X^comp = (1 - alpha * M) * X + alpha * M * C
    """
    img_arr = np.array(original_image.convert("RGB"), dtype=np.float32)
    img_h, img_w = img_arr.shape[:2]
    composite = img_arr.copy()

    masks_applied = 0
    for detection in damage_detections:
        if detection.mask is None:
            continue

        binary_mask = _binary_mask_in_image_space(detection, img_w, img_h)
        if binary_mask is None:
            continue

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


def _binary_mask_in_image_space(detection, img_w: int, img_h: int) -> np.ndarray | None:  # type: ignore[type-arg, no-untyped-def]
    """Return a ``(img_h, img_w)`` binary mask aligned to the original image,
    or ``None`` if the detection should be skipped.

    Split out of :func:`compose` so each placement strategy (full-image
    fallback vs. crop-local blit) is easy to reason about and test in
    isolation.
    """

    raw = np.array(detection.mask, dtype=np.float32)
    crop_box = getattr(detection, "crop_box_pixels", None)

    if crop_box is None or crop_box == (0, 0, 0, 0):
        # Legacy / whole-image mask: resize to frame and threshold. This
        # preserves the behaviour verified by ``test_composer*`` and is
        # the right thing for masks already emitted in original-image
        # coordinates (e.g. parts-level masks, tests).
        if raw.shape != (img_h, img_w):
            mask_img = Image.fromarray((raw * 255).astype(np.uint8)).resize((img_w, img_h), Image.NEAREST)
            raw = np.array(mask_img, dtype=np.float32) / 255.0
        return (raw >= MASK_THRESHOLD).astype(np.float32)

    x1, y1, x2, y2 = (int(v) for v in crop_box)
    # Clamp to frame — a malformed/out-of-frame crop box would otherwise
    # raise on the slice assignment below.
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        logger.warning(
            "Composer: skipping mask with zero-area crop_box_pixels={} for damage_type={}",
            crop_box,
            getattr(detection, "damage_type", "?"),
        )
        return None

    # YOLO-seg masks can come at model-output resolution (e.g. 160x160)
    # rather than at crop resolution. Resize to exactly the crop area in
    # original-image pixels so the placement is unambiguous.
    if raw.shape != (box_h, box_w):
        mask_img = Image.fromarray((raw * 255).astype(np.uint8)).resize((box_w, box_h), Image.NEAREST)
        raw = np.array(mask_img, dtype=np.float32) / 255.0

    binary_crop = (raw >= MASK_THRESHOLD).astype(np.float32)
    full = np.zeros((img_h, img_w), dtype=np.float32)
    full[y1:y2, x1:x2] = binary_crop
    return full
