from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

EXCLUDED_PARTS = {"headlight_left", "headlight_right"}


@dataclass
class PartDetection:
    part_type: str
    confidence: float
    bbox: list[float]
    mask: Any | None


class PartsDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.7) -> None:
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._model: Any | None = None

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore[import-untyped]

        self._model = YOLO(self._model_path)
        logger.info("Parts model loaded from {}", self._model_path)

    def predict(self, image_path_or_array: Any) -> list[PartDetection]:
        if self._model is None:
            raise RuntimeError("PartsDetector not loaded; call load() first")

        results = self._model(image_path_or_array, verbose=False)
        detections: list[PartDetection] = []

        for result in results:
            boxes = result.boxes
            masks = result.masks if result.masks is not None else None
            names = result.names

            for i, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                part_type = names[class_id]

                if confidence < self._threshold:
                    continue
                if part_type in EXCLUDED_PARTS:
                    continue

                bbox = box.xywhn[0].tolist()
                mask = masks.data[i].cpu().numpy() if masks is not None else None

                detections.append(PartDetection(part_type=part_type, confidence=confidence, bbox=bbox, mask=mask))

        logger.debug("PartsDetector found {} detections", len(detections))
        return detections
