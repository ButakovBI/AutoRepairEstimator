from __future__ import annotations

import io
from dataclasses import dataclass

from loguru import logger
from PIL import Image

from auto_repair_estimator.ml_worker.inference.parts_detector import PartDetection


@dataclass
class Crop:
    part_type: str
    confidence: float
    bbox: list[float]
    crop_bytes: bytes
    crop_key: str


def crop_parts(
    original_image: Image.Image,
    detections: list[PartDetection],
    request_id: str,
    bucket: str,
    excluded_parts: frozenset[str] | set[str] | None = None,
) -> list[Crop]:
    """Crop parts from ``original_image`` according to YOLO-normalised ``xywhn`` bboxes.

    Degenerate inputs (malformed bbox, fully-out-of-frame bbox, zero-area crop
    after clamping) are **skipped with a warning** rather than raising or
    emitting unreadable JPEG bytes. Downstream consumers (damage detector,
    S3 preview, composer) depend on every returned ``Crop.crop_bytes`` being
    a valid JPEG with positive area.
    """
    crops: list[Crop] = []
    width, height = original_image.size
    skip = excluded_parts or frozenset()

    for i, detection in enumerate(detections):
        if detection.part_type in skip:
            continue

        if len(detection.bbox) != 4:
            logger.warning(
                "crop_parts: skipping detection {} with malformed bbox (expected 4 values, got {}): part_type={}",
                i,
                len(detection.bbox),
                detection.part_type,
            )
            continue

        x_c, y_c, w, h = detection.bbox
        x1 = int((x_c - w / 2) * width)
        y1 = int((y_c - h / 2) * height)
        x2 = int((x_c + w / 2) * width)
        y2 = int((y_c + h / 2) * height)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "crop_parts: skipping detection {} with zero-area clamped bbox: part_type={} bbox={} clamped=({},{},{},{})",
                i,
                detection.part_type,
                detection.bbox,
                x1,
                y1,
                x2,
                y2,
            )
            continue

        cropped = original_image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=90)
        crop_bytes = buf.getvalue()
        crop_key = f"{bucket}/{request_id}_part_{i}_{detection.part_type}.jpg"

        crops.append(
            Crop(
                part_type=detection.part_type,
                confidence=detection.confidence,
                bbox=detection.bbox,
                crop_bytes=crop_bytes,
                crop_key=crop_key,
            )
        )

    return crops
