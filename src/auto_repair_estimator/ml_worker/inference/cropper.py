from __future__ import annotations

import io
from dataclasses import dataclass

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
    original_image: Image.Image, detections: list[PartDetection], request_id: str, bucket: str
) -> list[Crop]:
    crops: list[Crop] = []
    width, height = original_image.size

    for i, detection in enumerate(detections):
        x_c, y_c, w, h = detection.bbox
        x1 = int((x_c - w / 2) * width)
        y1 = int((y_c - h / 2) * height)
        x2 = int((x_c + w / 2) * width)
        y2 = int((y_c + h / 2) * height)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

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
