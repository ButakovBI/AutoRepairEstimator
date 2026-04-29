"""QA robustness tests for ``crop_parts``.

``PartsDetector`` in production returns ``bbox`` values in YOLO-normalised
``xywhn`` format, which is expected to satisfy ``0 <= x_c, y_c <= 1`` and
``w, h > 0``. However, nothing downstream enforces this contract:

    * Corrupt serialised detections (e.g. replayed from Kafka after an ML
      upgrade) may carry malformed bboxes.
    * A detector misbehaving under OOM or numerical overflow can emit
      centre coordinates outside [0, 1].
    * ``bbox`` is a plain ``list[float]`` so nothing prevents shorter
      sequences from reaching the cropper.

The cropper must degrade gracefully in these situations — either by
skipping the bad detection or by producing bytes that still decode as a
valid image. Today it does neither: a bbox fully outside the image yields
a zero-sized PIL crop, and a bbox with fewer than four elements raises a
raw ``ValueError`` on unpacking that bubbles up to the ML worker loop.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from auto_repair_estimator.ml_worker.inference.cropper import crop_parts
from auto_repair_estimator.ml_worker.inference.parts_detector import PartDetection


# 800x600 chosen arbitrarily; the specific pixel dimensions don't matter,
# only that they differ so any silent swapping of axes is exposed.
_IMG_W, _IMG_H = 800, 600


def _image() -> Image.Image:
    return Image.new("RGB", (_IMG_W, _IMG_H), color=(10, 20, 30))


def _detection(bbox: list[float], part_type: str = "hood") -> PartDetection:
    return PartDetection(part_type=part_type, confidence=0.95, bbox=bbox, mask=None)


def test_bbox_entirely_outside_image_does_not_emit_invalid_jpeg() -> None:
    """A centre at (2.0, 2.0) sits beyond the right/bottom edge.

    Clamping will produce a zero-area crop. Either the cropper skips this
    detection or it returns bytes that Pillow can decode back. Returning a
    ``Crop`` whose ``crop_bytes`` cannot be re-opened would break every
    downstream S3/URL consumer (damage detector, composer, S3 preview).
    """
    img = _image()
    # centre at (2.0, 2.0) puts the entire bbox beyond the image frame
    detection = _detection(bbox=[2.0, 2.0, 0.1, 0.1])

    crops = crop_parts(img, [detection], "req-1", "crops")

    if not crops:
        # Acceptable: cropper skipped the degenerate detection entirely.
        return

    assert len(crops) == 1
    try:
        with Image.open(io.BytesIO(crops[0].crop_bytes)) as decoded:
            decoded.load()
            w, h = decoded.size
    except Exception as exc:  # pragma: no cover - defensive, descriptive failure
        pytest.fail(
            f"Cropper returned bytes that Pillow cannot decode for an out-of-frame bbox: {exc}"
        )
    assert w > 0 and h > 0, (
        f"Cropper returned a zero-area image ({w}x{h}) for an out-of-frame bbox. "
        "Downstream consumers will fail on an empty crop."
    )


def test_bbox_with_fewer_than_four_values_is_handled_gracefully() -> None:
    """A malformed detection must not crash the whole inference loop.

    The expected behaviour is *either* a skipped detection (returned list
    shorter than the input) *or* a clearly named/caught exception that the
    caller can handle. A raw ``ValueError: not enough values to unpack``
    bubbling out of an internal tuple unpack is not acceptable for a
    production boundary that consumes untrusted/serialised data.
    """
    img = _image()
    good = _detection(bbox=[0.5, 0.5, 0.3, 0.3], part_type="bumper")
    malformed = _detection(bbox=[0.5, 0.5], part_type="hood")

    try:
        crops = crop_parts(img, [malformed, good], "req-1", "crops")
    except ValueError as exc:
        msg = str(exc)
        # The spec for robust input handling: if the cropper chooses to raise
        # rather than skip, the message must identify the bad input so an
        # operator can locate it — a raw unpack error does not.
        assert "bbox" in msg.lower() or "detection" in msg.lower(), (
            f"crop_parts raised an unhelpful ValueError on malformed bbox: {msg!r}. "
            "A production-ready cropper should skip or raise a descriptive error."
        )
        return

    kept_part_types = [c.part_type for c in crops]
    # If we got here, the cropper did not raise. It must then have skipped
    # the malformed detection but kept the valid one.
    assert "bumper" in kept_part_types, "Valid detection was lost alongside the malformed one."
    assert "hood" not in kept_part_types, (
        "Malformed detection silently produced a crop — the cropper should have skipped it."
    )


def test_crop_key_matches_the_detection_it_was_produced_from() -> None:
    """Regression guard.

    ``crop_key`` is indexed by the enumeration index ``i`` of the input list.
    If future refactors skip some detections mid-iteration, the saved crop
    keys and the returned ``Crop.part_type`` fields must still agree, or
    downstream code that keys parts by filename will mis-attribute damages.
    """
    img = _image()
    detections = [
        _detection(bbox=[0.25, 0.25, 0.2, 0.2], part_type="hood"),
        _detection(bbox=[0.75, 0.75, 0.2, 0.2], part_type="bumper"),
    ]
    crops = crop_parts(img, detections, "req-1", "crops")

    for crop in crops:
        # The filename substring for part_type must match the emitted field.
        assert f"_{crop.part_type}." in crop.crop_key, (
            f"Crop key {crop.crop_key!r} does not encode its own part_type {crop.part_type!r}; "
            "damages attributed by filename would be mis-joined."
        )
