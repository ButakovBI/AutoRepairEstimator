from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from auto_repair_estimator.backend.domain.value_objects.ml_thresholds import (
    PARTS_CONFIDENCE_THRESHOLD,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import PartType

# See note in damage_detector.py — we keep the enum as the single source of
# truth for which class strings are valid downstream.
_SUPPORTED_PART_NAMES: frozenset[str] = frozenset(member.value for member in PartType)


@dataclass
class PartDetection:
    part_type: str
    confidence: float
    bbox: list[float]
    mask: Any | None


class PartsDetector:
    # The default comes from the shared SSOT module so unit tests, CLI
    # scripts and ad-hoc invocations never silently drift from what the
    # worker runs in prod. Operators can still override via the
    # ``MLWorkerConfig`` env knob — that value is threaded in by
    # ``ml_worker.main.process_request``.
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = PARTS_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._model: Any | None = None

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        self._model = YOLO(self._model_path)
        self._assert_model_classes_match_enum()
        logger.info("Parts model loaded from {}", self._model_path)

    def _assert_model_classes_match_enum(self) -> None:
        names_attr = getattr(self._model, "names", None)
        if names_attr is None:
            logger.warning("Parts model has no `names` metadata; class filtering disabled")
            return

        model_names = set(names_attr.values()) if isinstance(names_attr, dict) else set(names_attr)
        unknown = model_names - _SUPPORTED_PART_NAMES
        missing = _SUPPORTED_PART_NAMES - model_names

        if unknown:
            logger.warning(
                "Parts model exposes classes not in PartType enum; these will be dropped: {}",
                sorted(unknown),
            )
        if missing:
            logger.warning(
                "Parts model is missing these PartType classes (they will never be detected): {}",
                sorted(missing),
            )
        if not (model_names & _SUPPORTED_PART_NAMES):
            logger.error(
                "Parts model classes {} share NO overlap with PartType enum {} — wrong weights?",
                sorted(model_names),
                sorted(_SUPPORTED_PART_NAMES),
            )

    def predict(
        self,
        image_path_or_array: Any,
        request_id: str | None = None,
    ) -> list[PartDetection]:
        """Run the parts model and return the accepted detections.

        The ``request_id`` argument is purely for log correlation —
        callers that use this detector outside the Kafka pipeline (e.g.
        unit tests, CLI scripts) can pass ``None`` and the log lines
        will still emit with ``request=-``. Having request_id woven
        through the diagnostic logs is what makes it possible to read
        ``docker logs auto-repair-ml-worker | grep <request_id>`` and
        see exactly what the model saw for a single user submission.
        """
        if self._model is None:
            raise RuntimeError("PartsDetector not loaded; call load() first")

        req_tag = request_id or "-"
        results = self._model(image_path_or_array, verbose=False)
        detections: list[PartDetection] = []
        dropped_unknown = 0
        # Raw tallies used to log one structured summary per request.
        # Each entry: (class_label, confidence, verdict). Verdict values:
        # "ACCEPTED", "REJECTED_LOW_CONF", "REJECTED_UNKNOWN_CLASS".
        raw_tally: list[tuple[str, float, str]] = []

        for result in results:
            boxes = result.boxes
            masks = result.masks if result.masks is not None else None
            names = result.names

            for i, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                part_type = names[class_id]

                if confidence < self._threshold:
                    raw_tally.append((part_type, confidence, "REJECTED_LOW_CONF"))
                    continue

                if part_type not in _SUPPORTED_PART_NAMES:
                    dropped_unknown += 1
                    raw_tally.append((part_type, confidence, "REJECTED_UNKNOWN_CLASS"))
                    continue

                bbox = box.xywhn[0].tolist()
                mask = masks.data[i].cpu().numpy() if masks is not None else None

                detections.append(PartDetection(part_type=part_type, confidence=confidence, bbox=bbox, mask=mask))
                raw_tally.append((part_type, confidence, "ACCEPTED"))

        if dropped_unknown:
            logger.warning(
                "PartsDetector[request={}] dropped {} detections with classes outside PartType enum",
                req_tag,
                dropped_unknown,
            )

        # Always log the full raw model output at INFO level: "the
        # bumper wasn't detected" is almost always a low-confidence
        # issue, and we want that visible in `docker logs` without
        # having to bump the log level. Sorted by descending confidence
        # so the highest-scoring rejections (the ones closest to the
        # threshold) appear first and are easiest to eyeball.
        summary = sorted(raw_tally, key=lambda entry: -entry[1])
        summary_str = ", ".join(f"{cls}@{conf:.2f}:{verdict}" for cls, conf, verdict in summary) or "<none>"
        logger.info(
            "PartsDetector[request={}] raw={} accepted={} threshold={:.2f} | {}",
            req_tag,
            len(raw_tally),
            len(detections),
            self._threshold,
            summary_str,
        )
        return detections
