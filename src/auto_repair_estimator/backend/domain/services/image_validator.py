"""Validation rules for user-uploaded photos.

Kept in the domain layer as a pure function so it can be exercised in unit
tests without any infrastructure. The rules are deliberately simple and
aligned with what the ML pipeline is able to digest (see
``ml_worker.inference.preprocessor``): JPEG/PNG/WEBP, up to 10 MB, with the
shorter side at least 320 px.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Final

from PIL import Image

SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset({"JPEG", "PNG", "WEBP"})
MAX_IMAGE_BYTES: Final[int] = 10 * 1024 * 1024
MIN_SIDE_PX: Final[int] = 320


@dataclass(frozen=True)
class ImageMetadata:
    format: str
    width: int
    height: int
    size_bytes: int


def validate_image_bytes(data: bytes) -> ImageMetadata:
    """Validate raw image bytes.

    Raises ``ValueError`` with a user-friendly message if the image does not
    meet the rules; returns parsed metadata on success.
    """

    size = len(data)
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large: {size} bytes (max {MAX_IMAGE_BYTES})")
    if size == 0:
        raise ValueError("image is empty")

    try:
        with Image.open(io.BytesIO(data)) as img:
            img_format = img.format or ""
            width, height = img.size
    except Exception as exc:
        raise ValueError(f"image cannot be decoded: {exc}") from exc

    if img_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported image format: {img_format!r} (supported: {sorted(SUPPORTED_FORMATS)})")

    if min(width, height) < MIN_SIDE_PX:
        raise ValueError(f"image too small: {width}x{height} (min {MIN_SIDE_PX}px on shorter side)")

    return ImageMetadata(format=img_format, width=width, height=height, size_bytes=size)
