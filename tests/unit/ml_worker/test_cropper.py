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


class TestCropperDiagnosticSummary:
    """The summary log is the operator's answer to "which parts went
    into damage detection?". It MUST list every accepted crop with its
    index + part_type + confidence, because those indices are the
    correlation key with DamageDetector log lines downstream.
    """

    def test_summary_lists_accepted_crops_with_indices_and_confidence(
        self, caplog
    ) -> None:
        import logging

        from loguru import logger as loguru_logger

        handler_id = loguru_logger.add(
            caplog.handler, level="INFO", format="{message}"
        )
        caplog.set_level(logging.INFO)

        try:
            img = _make_image()
            detections = [
                _detection("hood", confidence=0.91),
                _detection("bumper", confidence=0.84),
            ]
            crop_parts(img, detections, "req-summary", "crops")
        finally:
            loguru_logger.remove(handler_id)

        all_text = " ".join(record.message for record in caplog.records)
        assert "crop_parts[request=req-summary]" in all_text
        assert "accepted=2" in all_text
        # Each accepted crop must appear with its index, part_type, conf —
        # the indices here are the same ones DamageDetector logs use.
        assert "crop=0:hood@0.91" in all_text
        assert "crop=1:bumper@0.84" in all_text

    def test_summary_reports_config_excluded_parts_distinctly_from_degenerate(
        self, caplog
    ) -> None:
        """Operator needs to know *why* a detection didn't become a
        crop — excluded-by-config and degenerate-bbox have very
        different fixes, so they must be logged in separate buckets.
        """

        import logging

        from loguru import logger as loguru_logger

        handler_id = loguru_logger.add(
            caplog.handler, level="INFO", format="{message}"
        )
        caplog.set_level(logging.INFO)

        try:
            img = _make_image()
            detections = [
                _detection("hood", confidence=0.91),
                _detection("headlight", confidence=0.77),  # excluded by config
                PartDetection(
                    part_type="bumper",
                    confidence=0.81,
                    bbox=[0.5, 0.5],  # malformed: not 4 values
                    mask=None,
                ),
            ]
            crop_parts(
                img,
                detections,
                "req-3",
                "crops",
                excluded_parts={"headlight"},
            )
        finally:
            loguru_logger.remove(handler_id)

        all_text = " ".join(record.message for record in caplog.records)
        assert "accepted=1" in all_text
        assert "excluded=1" in all_text
        assert "degenerate=1" in all_text
        assert "excluded_by_config=[headlight@0.77]" in all_text
        assert "bumper@0.81:malformed_bbox" in all_text
