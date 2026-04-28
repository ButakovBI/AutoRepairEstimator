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
    # Absolute pixel coordinates of the crop inside the original image
    # (x1, y1, x2, y2). The damage detector runs on the cropped bytes and
    # therefore emits masks in *crop-local* pixel space; the composer
    # needs these original-image coordinates to place each mask back at
    # the correct spot on the full photo. Without this, masks stretch
    # across the whole frame (the regression the user reported — a door
    # damage mask leaking across the entire car body).
    crop_box_pixels: tuple[int, int, int, int] = (0, 0, 0, 0)


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

    Emits a single structured INFO summary at the end of the call:

        crop_parts[request=...] accepted=... excluded=... degenerate=...
          | crop=0:door@0.91 crop=1:bumper@0.84 ...

    This line is the canonical answer to "which parts went into damage
    detection?" — the ``crop=<i>`` indices here are exactly the
    ``crop=<i>`` indices that appear in later ``DamageDetector[...]``
    log lines, so an operator can correlate one-for-one.
    """
    crops: list[Crop] = []
    width, height = original_image.size
    skip = excluded_parts or frozenset()

    excluded_by_config: list[tuple[str, float]] = []
    degenerate: list[tuple[str, float, str]] = []  # (part_type, confidence, reason)

    for i, detection in enumerate(detections):
        if detection.part_type in skip:
            excluded_by_config.append((detection.part_type, detection.confidence))
            continue

        if len(detection.bbox) != 4:
            logger.warning(
                "crop_parts: skipping detection {} with malformed bbox (expected 4 values, got {}): part_type={}",
                i,
                len(detection.bbox),
                detection.part_type,
            )
            degenerate.append((detection.part_type, detection.confidence, "malformed_bbox"))
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
            degenerate.append((detection.part_type, detection.confidence, "zero_area"))
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
                crop_box_pixels=(x1, y1, x2, y2),
            )
        )

    accepted_summary = (
        " ".join(f"crop={idx}:{c.part_type}@{c.confidence:.2f}" for idx, c in enumerate(crops)) or "<none>"
    )
    extras: list[str] = []
    if excluded_by_config:
        extras.append("excluded_by_config=[" + ", ".join(f"{pt}@{conf:.2f}" for pt, conf in excluded_by_config) + "]")
    if degenerate:
        extras.append("degenerate=[" + ", ".join(f"{pt}@{conf:.2f}:{reason}" for pt, conf, reason in degenerate) + "]")
    extras_str = (" " + " ".join(extras)) if extras else ""

    logger.info(
        "crop_parts[request={}] accepted={} excluded={} degenerate={} | {}{}",
        request_id,
        len(crops),
        len(excluded_by_config),
        len(degenerate),
        accepted_summary,
        extras_str,
    )

    return crops
