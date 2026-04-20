"""QA: image validator must not be fooled by format trickery.

Attack surface: the bot receives bytes via ``httpx`` from a URL returned
by the VK API. An attacker (or simply a broken client library) could
deliver:

* an animated GIF with ``Content-Type: image/jpeg``,
* a ZIP or other non-image payload with a JPEG magic number prepended,
* a legitimate JPEG that happens to be 319 px on the shorter side (one
  pixel below the pipeline's minimum),
* a truncated JPEG that starts parsing but fails halfway,
* a massive (>10 MB) but otherwise valid JPEG.

``validate_image_bytes`` is the single chokepoint before the ML pipeline;
it must reject all of these with a ``ValueError`` that carries a
user-friendly message. It must never raise an unexpected exception type
or accept the bytes.

The tests use tiny in-memory PIL images so we don't rely on fixtures or
real network traffic, and each sample encodes the specific attack in
pytest IDs.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from auto_repair_estimator.backend.domain.services.image_validator import (
    MAX_IMAGE_BYTES,
    MIN_SIDE_PX,
    validate_image_bytes,
)


def _encode(fmt: str, width: int, height: int, color=(200, 200, 200)) -> bytes:
    """Encode a solid-colour image in the requested format."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format=fmt)
    return buf.getvalue()


def test_animated_gif_is_rejected_even_when_delivered_as_jpeg_url() -> None:
    """PIL identifies the payload by magic bytes, not by the URL extension,
    so a GIF arriving from a ``.jpg`` URL still gets classified as GIF —
    which is not in SUPPORTED_FORMATS."""

    # Minimal 400x400 solid GIF (animated or not, PIL reports "GIF").
    buf = io.BytesIO()
    Image.new("RGB", (400, 400), (10, 10, 10)).save(buf, format="GIF")
    gif_bytes = buf.getvalue()

    with pytest.raises(ValueError, match="unsupported image format"):
        validate_image_bytes(gif_bytes)


def test_jpeg_just_below_min_side_is_rejected() -> None:
    """MIN_SIDE_PX boundary — an off-by-one error here would let tiny VK
    thumbnails leak into the ML pipeline and cause silent detection
    failures."""

    # One pixel below the minimum on the shorter side.
    too_small = _encode("JPEG", MIN_SIDE_PX - 1, MIN_SIDE_PX + 100)

    with pytest.raises(ValueError, match="image too small"):
        validate_image_bytes(too_small)


def test_jpeg_exactly_at_min_side_is_accepted() -> None:
    """The complement: exactly at the boundary must pass to avoid
    frustrating users who shoot square photos from older phones."""

    ok = _encode("JPEG", MIN_SIDE_PX, MIN_SIDE_PX)
    meta = validate_image_bytes(ok)
    assert meta.format == "JPEG"
    assert meta.width == MIN_SIDE_PX
    assert meta.height == MIN_SIDE_PX


def test_oversized_payload_is_rejected_before_decode() -> None:
    """Over-size rejection must happen BEFORE PIL.Image.open — otherwise a
    decompression-bomb image could exhaust RAM in the backend."""

    # Fill with JPEG magic bytes so if the implementation tries to decode
    # first and rejects later, it fails for "wrong reason" and the match
    # string would catch that regression.
    oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (MAX_IMAGE_BYTES + 10)
    with pytest.raises(ValueError, match="image too large"):
        validate_image_bytes(oversized)


def test_truncated_jpeg_is_rejected_with_decode_error() -> None:
    """PIL raises when it can't parse the stream; the validator must wrap
    that in a ``ValueError`` with a user-friendly prefix rather than
    letting the raw exception bubble through."""

    full = _encode("JPEG", 500, 500)
    # Cut off the last 90% — JPEG header stays but scan data is missing.
    truncated = full[: max(10, len(full) // 10)]
    with pytest.raises(ValueError, match="image cannot be decoded"):
        validate_image_bytes(truncated)


def test_random_bytes_are_rejected_not_mistaken_for_image() -> None:
    """Adversary sends arbitrary bytes that are valid UTF-8; PIL must not
    confuse this for a valid image."""

    junk = b"This is not an image and must be rejected by validate_image_bytes." * 20
    with pytest.raises(ValueError):
        validate_image_bytes(junk)


def test_empty_bytes_raise_explicit_empty_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_image_bytes(b"")
