"""Render raw YOLO-seg mask arrays into diagnostic PNG bytes.

These PNGs are persisted to MinIO alongside crops and composites so an
operator can open each mask individually and verify what the model
actually saw. They're **not** consumed by any downstream service — they
exist purely for human inspection via the MinIO console.

Two helpers:

* :func:`mask_to_grayscale_png_bytes` — the raw binary mask as a
  black-and-white PNG. Compact, useful for checking coverage shape.
* :func:`mask_overlay_png_bytes` — the mask blended on top of the
  original crop in a semi-transparent colour, for a "what the model
  highlighted on this detail" preview.

Both accept the raw YOLO mask (float array in ``[0, 1]`` or ``{0, 1}``
at arbitrary resolution) and resize to the requested output size using
nearest-neighbour (matching the main composer's approach — we don't
want anti-aliasing to blur mask boundaries in diagnostics).
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

_MASK_THRESHOLD = 0.5


def mask_to_grayscale_png_bytes(
    mask: Any,
    target_size: tuple[int, int] | None = None,
) -> bytes:
    """Render ``mask`` as a single-channel ``L``-mode PNG.

    ``target_size`` is ``(width, height)``; when provided, the mask is
    resized to it before thresholding so the saved file lines up pixel-
    for-pixel with the crop or original image the operator is comparing
    against. When omitted, the mask is saved at its native resolution
    (typically YOLO's 160×160 output).
    """

    arr = np.array(mask, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if target_size is not None:
        target_w, target_h = target_size
        if (arr.shape[1], arr.shape[0]) != (target_w, target_h):
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.resize((target_w, target_h), Image.NEAREST)
            arr = np.array(img, dtype=np.float32) / 255.0

    binary = (arr >= _MASK_THRESHOLD).astype(np.uint8) * 255
    out = Image.fromarray(binary, mode="L")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def mask_overlay_png_bytes(
    base_image: Image.Image,
    mask: Any,
    color: tuple[int, int, int],
    alpha: float = 0.5,
) -> bytes:
    """Render ``mask`` as a semi-transparent colour overlay on ``base_image``.

    The resulting PNG is the exact size of ``base_image`` (so for a
    part overlay, pass the crop; for a damage overlay, pass the crop
    too since damage masks are in crop-local space). Useful for quickly
    eyeballing "did the model highlight the right region?".
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    rgb = np.array(base_image.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]

    arr = np.array(mask, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (h, w):
        resized = Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        arr = np.array(resized, dtype=np.float32) / 255.0
    binary = (arr >= _MASK_THRESHOLD).astype(np.float32)

    colour_arr = np.array(color, dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = (1.0 - alpha * binary) * rgb[:, :, c] + alpha * binary * colour_arr[c]

    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
