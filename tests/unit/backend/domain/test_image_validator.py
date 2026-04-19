"""Adversarial tests for ``validate_image_bytes``.

The validator sits on the HTTP boundary between the VK bot (attacker surface)
and the ML worker (trust boundary). The ML pipeline assumes JPEG/PNG/WEBP,
a shorter side of at least 320 px, and a payload no larger than 10 MB. Anything
else must be rejected with ``ValueError`` *before* it reaches Kafka — otherwise
Pillow / YOLO will crash inside the worker and stall the inference queue.

These tests deliberately probe the boundaries and common adversarial inputs:

1. empty bytes, 1-byte garbage, random non-image bytes — must not decode;
2. unsupported image formats (GIF, BMP, TIFF) that Pillow can open — must be
   rejected by the allow-list check;
3. images that are exactly at / just under the minimum side — boundary of the
   dimension check;
4. an image that is within the size limit but whose shorter side is 319 px —
   fails dimension rule;
5. a payload one byte over ``MAX_IMAGE_BYTES`` — fails size rule *before*
   decoding (fast-path, protects the worker from decoding huge uploads).
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from auto_repair_estimator.backend.domain.services.image_validator import (
    MAX_IMAGE_BYTES,
    MIN_SIDE_PX,
    ImageMetadata,
    validate_image_bytes,
)


def _encode(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=90)
    return buf.getvalue()


def _valid_jpeg(width: int = MIN_SIDE_PX, height: int = MIN_SIDE_PX) -> bytes:
    # MIN_SIDE_PX is the smallest accepted shorter-side value, so a square at
    # that size is the minimum legal happy-path input.
    return _encode(Image.new("RGB", (width, height), color=(10, 20, 30)), "JPEG")


class TestHappyPath:
    def test_minimum_valid_jpeg_returns_metadata_with_exact_dimensions(self) -> None:
        data = _valid_jpeg(MIN_SIDE_PX, MIN_SIDE_PX)

        meta = validate_image_bytes(data)

        assert isinstance(meta, ImageMetadata)
        assert meta.format == "JPEG"
        assert meta.width == MIN_SIDE_PX
        assert meta.height == MIN_SIDE_PX
        assert meta.size_bytes == len(data)

    def test_png_is_accepted(self) -> None:
        data = _encode(Image.new("RGB", (400, 400), color=(0, 0, 0)), "PNG")

        meta = validate_image_bytes(data)

        assert meta.format == "PNG"

    def test_webp_is_accepted(self) -> None:
        data = _encode(Image.new("RGB", (400, 400), color=(1, 2, 3)), "WEBP")

        meta = validate_image_bytes(data)

        assert meta.format == "WEBP"


class TestSizeRules:
    def test_empty_bytes_are_rejected_with_empty_message(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_image_bytes(b"")

    def test_one_byte_payload_is_rejected_as_non_image(self) -> None:
        # 1 byte passes the "not empty" check but cannot be decoded; this
        # protects against callers that trim or corrupt the upload.
        with pytest.raises(ValueError, match="cannot be decoded"):
            validate_image_bytes(b"\x00")

    def test_payload_exceeding_max_size_is_rejected_before_decode(self) -> None:
        # One byte over the limit. Using b"\x00" padding is cheaper than
        # generating a real oversized image and still exercises the size
        # short-circuit (which runs BEFORE Pillow touches the bytes).
        oversized = b"\x00" * (MAX_IMAGE_BYTES + 1)

        with pytest.raises(ValueError, match="too large"):
            validate_image_bytes(oversized)

    def test_payload_at_exactly_max_size_would_still_require_decodability(self) -> None:
        # Boundary: ``size == MAX_IMAGE_BYTES`` passes the size check; since
        # our bytes are non-image they must then fail at the decode step.
        not_an_image = b"\x00" * MAX_IMAGE_BYTES

        with pytest.raises(ValueError, match="cannot be decoded"):
            validate_image_bytes(not_an_image)


class TestFormatRules:
    def test_random_bytes_fail_with_decode_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be decoded"):
            validate_image_bytes(b"not-an-image-\x00\x01\x02\x03\x04\xff" * 100)

    def test_gif_is_rejected_even_though_pillow_can_decode_it(self) -> None:
        # GIF is a legitimate Pillow-readable format, but intentionally
        # excluded (animations + palette issues would confuse YOLO).
        data = _encode(Image.new("P", (400, 400), color=0), "GIF")

        with pytest.raises(ValueError, match="unsupported image format"):
            validate_image_bytes(data)

    def test_bmp_is_rejected(self) -> None:
        data = _encode(Image.new("RGB", (400, 400), color=(0, 0, 0)), "BMP")

        with pytest.raises(ValueError, match="unsupported image format"):
            validate_image_bytes(data)

    def test_tiff_is_rejected(self) -> None:
        data = _encode(Image.new("RGB", (400, 400), color=(0, 0, 0)), "TIFF")

        with pytest.raises(ValueError, match="unsupported image format"):
            validate_image_bytes(data)


class TestDimensionRules:
    def test_image_just_under_min_side_is_rejected(self) -> None:
        # ``MIN_SIDE_PX - 1`` on the shorter side is the tightest failure case.
        data = _valid_jpeg(MIN_SIDE_PX - 1, MIN_SIDE_PX)

        with pytest.raises(ValueError, match="too small"):
            validate_image_bytes(data)

    def test_tall_but_narrow_image_is_rejected_on_shorter_side(self) -> None:
        # Tall portrait: 4000 px tall but 100 px wide — must still fail
        # because the rule is on the SHORTER side, not the longer one.
        data = _valid_jpeg(100, 4000)

        with pytest.raises(ValueError, match="too small"):
            validate_image_bytes(data)

    def test_image_at_exact_minimum_side_passes(self) -> None:
        # Boundary: exactly ``MIN_SIDE_PX`` must be accepted.
        data = _valid_jpeg(MIN_SIDE_PX, 4000)

        meta = validate_image_bytes(data)

        assert min(meta.width, meta.height) == MIN_SIDE_PX
