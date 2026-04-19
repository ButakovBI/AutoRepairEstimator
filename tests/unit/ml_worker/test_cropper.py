from __future__ import annotations

import io

from PIL import Image

from auto_repair_estimator.ml_worker.inference.cropper import crop_parts
from auto_repair_estimator.ml_worker.inference.parts_detector import PartDetection


def _make_image(width: int = 800, height: int = 600) -> Image.Image:
    return Image.new("RGB", (width, height), color=(100, 150, 200))


def _detection(part_type: str, confidence: float = 0.85) -> PartDetection:
    return PartDetection(
        part_type=part_type,
        confidence=confidence,
        bbox=[0.5, 0.5, 0.3, 0.3],
        mask=None,
    )


def test_crop_parts_returns_correct_count() -> None:
    img = _make_image()
    detections = [_detection("hood"), _detection("bumper")]
    crops = crop_parts(img, detections, "req-1", "crops")
    assert len(crops) == 2


def test_crop_parts_sets_correct_key_format() -> None:
    img = _make_image()
    detections = [_detection("hood")]
    crops = crop_parts(img, detections, "my-request", "crops")
    assert crops[0].crop_key.startswith("crops/my-request_part_0_hood")


def test_crop_parts_produces_valid_jpeg() -> None:
    img = _make_image()
    detections = [_detection("hood")]
    crops = crop_parts(img, detections, "req-1", "crops")
    crop_img = Image.open(io.BytesIO(crops[0].crop_bytes))
    assert crop_img.format == "JPEG"


def test_crop_parts_with_empty_detections_returns_empty() -> None:
    img = _make_image()
    crops = crop_parts(img, [], "req-1", "crops")
    assert len(crops) == 0


def test_crop_preserves_part_info() -> None:
    img = _make_image()
    detections = [_detection("front_fender", confidence=0.9)]
    crops = crop_parts(img, detections, "req-1", "crops")
    assert crops[0].part_type == "front_fender"
    assert abs(crops[0].confidence - 0.9) < 1e-6


def test_crop_parts_skips_excluded_parts() -> None:
    img = _make_image()
    detections = [_detection("hood"), _detection("headlight"), _detection("bumper")]
    crops = crop_parts(img, detections, "req-1", "crops", excluded_parts={"headlight"})
    kept = [c.part_type for c in crops]
    assert kept == ["hood", "bumper"]
