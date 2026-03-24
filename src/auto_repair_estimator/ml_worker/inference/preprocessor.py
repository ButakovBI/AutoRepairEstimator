from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from loguru import logger
from PIL import Image

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
INPUT_SIZE = 640


@dataclass
class PreprocessResult:
    original_image: Image.Image
    tensor: np.ndarray  # type: ignore[type-arg]
    original_size: tuple[int, int]


def preprocess(image_bytes: bytes, max_bytes: int = 10 * 1024 * 1024) -> PreprocessResult:
    if len(image_bytes) > max_bytes:
        raise ValueError(f"Image too large: {len(image_bytes)} bytes (max {max_bytes})")

    img = Image.open(io.BytesIO(image_bytes))

    if img.format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {img.format} (supported: {SUPPORTED_FORMATS})")

    original_size = img.size
    if min(original_size) < 320:
        raise ValueError(f"Image too small: {original_size} (min 320px)")

    img_rgb = img.convert("RGB")
    resized = img_rgb.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0

    logger.debug("Preprocessed image original_size={} tensor_shape={}", original_size, arr.shape)
    return PreprocessResult(original_image=img_rgb, tensor=arr, original_size=original_size)
