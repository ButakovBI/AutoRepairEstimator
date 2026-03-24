from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class DamageDetection:
    damage_type: str
    part_type: str
    confidence: float
    mask: Any | None
    mask_key: str | None = None


class DamageDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5) -> None:
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._model: Any | None = None

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore[import-untyped]

        self._model = YOLO(self._model_path)
        logger.info("Damages model loaded from {}", self._model_path)

    def predict(
        self, crop_bytes: bytes, part_type: str, request_id: str, crop_index: int, bucket: str
    ) -> list[DamageDetection]:
        if self._model is None:
            raise RuntimeError("DamageDetector not loaded; call load() first")

        from PIL import Image

        img = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
        results = self._model(img, verbose=False)
        detections: list[DamageDetection] = []

        for result in results:
            boxes = result.boxes
            masks = result.masks if result.masks is not None else None
            names = result.names

            for i, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                damage_type = names[class_id]

                if confidence < self._threshold:
                    continue

                mask = masks.data[i].cpu().numpy() if masks is not None else None
                mask_key = (
                    f"{bucket}/{request_id}_damage_{crop_index}_{i}_{damage_type}.png" if mask is not None else None
                )

                detections.append(
                    DamageDetection(
                        damage_type=damage_type,
                        part_type=part_type,
                        confidence=confidence,
                        mask=mask,
                        mask_key=mask_key,
                    )
                )

        logger.debug("DamageDetector found {} detections for part={}", len(detections), part_type)
        return detections
