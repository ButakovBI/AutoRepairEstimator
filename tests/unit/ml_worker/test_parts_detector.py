"""Unit tests for :class:`PartsDetector`.

Mirror of ``test_damage_detector.py`` for the parts model — same contract,
same filtering rules, different enum. Deduplicating the fake-YOLO scaffold
would add indirection without real value, so we keep each file self-contained
and easy to run in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from auto_repair_estimator.backend.domain.value_objects.request_enums import PartType
from auto_repair_estimator.ml_worker.inference.parts_detector import PartsDetector


@dataclass
class _FakeBox:
    conf: list[float]
    cls: list[int]
    # xywhn is read by the parts detector (bbox for the cropper), so the
    # fake needs to expose it. One box → one row of 4 normalised floats.
    xywhn: list[list[float]]


@dataclass
class _FakeResult:
    boxes: list[_FakeBox]
    names: dict[int, str]
    masks: Any | None = None


class _FakeYOLO:
    def __init__(self, names: dict[int, str], results: list[_FakeResult]) -> None:
        self.names = names
        self._results = results

    def __call__(self, _image: Any, verbose: bool = False) -> list[_FakeResult]:  # noqa: ARG002
        return self._results


def _box(conf: float, cls: int) -> _FakeBox:
    # A minimal but valid xywhn row: centre at (0.5, 0.5), size 0.4×0.4.
    # We don't test geometry here — just that the detector faithfully passes
    # the bbox through so the cropper has something to work with.
    class _XY:
        def __init__(self, data: list[float]) -> None:
            self._data = data

        def tolist(self) -> list[float]:
            return self._data

    return _FakeBox(conf=[conf], cls=[cls], xywhn=[_XY([0.5, 0.5, 0.4, 0.4])])  # type: ignore[arg-type]


class TestPartsDetectorPredictFiltering:
    def test_drops_detections_with_unknown_class_name(self) -> None:
        # Analogous to DamageDetector — class names outside PartType must not
        # reach the cropper (otherwise the crop S3 key and downstream storage
        # would contain unexpected enum values).
        detector = PartsDetector(model_path="unused", confidence_threshold=0.5)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: PartType.HOOD.value, 1: "spoiler"},
            results=[
                _FakeResult(
                    boxes=[
                        _box(0.95, 0),  # legit hood
                        _box(0.9, 1),  # unknown "spoiler" — must drop
                    ],
                    names={0: PartType.HOOD.value, 1: "spoiler"},
                    masks=None,
                )
            ],
        )

        detections = detector.predict(image_path_or_array=np.zeros((10, 10, 3), dtype=np.uint8))

        assert len(detections) == 1
        assert detections[0].part_type == PartType.HOOD.value

    def test_drops_detections_below_threshold(self) -> None:
        # Parts model runs at the higher default threshold (0.7); a 0.65
        # detection of a known class must still be rejected.
        detector = PartsDetector(model_path="unused", confidence_threshold=0.7)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: PartType.DOOR.value},
            results=[
                _FakeResult(
                    boxes=[_box(0.65, 0)],
                    names={0: PartType.DOOR.value},
                    masks=None,
                )
            ],
        )

        assert detector.predict(image_path_or_array=np.zeros((10, 10, 3), dtype=np.uint8)) == []

    def test_raises_when_not_loaded(self) -> None:
        detector = PartsDetector(model_path="unused")
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict(image_path_or_array=np.zeros((10, 10, 3), dtype=np.uint8))


class TestPartsDetectorDiagnosticLogging:
    """The raw-output summary log is the operator's primary tool for
    debugging "why didn't the model see that part?" — it MUST include
    below-threshold detections with their verdict. These tests pin that
    contract so nobody silently downgrades the line to DEBUG or drops
    the rejected entries."""

    def test_summary_log_includes_below_threshold_detection_with_verdict(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from loguru import logger as loguru_logger

        # Loguru → stdlib bridge so caplog captures the INFO line.
        handler_id = loguru_logger.add(
            caplog.handler, level="INFO", format="{message}"
        )
        caplog.set_level(logging.INFO)

        try:
            detector = PartsDetector(model_path="unused", confidence_threshold=0.7)
            detector._model = _FakeYOLO(  # type: ignore[attr-defined]
                names={0: PartType.BUMPER.value, 1: PartType.HOOD.value},
                results=[
                    _FakeResult(
                        boxes=[
                            # Below-threshold bumper — this is the kind of
                            # detection the user's question needs visibility
                            # on.
                            _box(0.62, 0),
                            # Accepted hood for contrast.
                            _box(0.95, 1),
                        ],
                        names={0: PartType.BUMPER.value, 1: PartType.HOOD.value},
                        masks=None,
                    )
                ],
            )

            detector.predict(
                image_path_or_array=np.zeros((10, 10, 3), dtype=np.uint8),
                request_id="req-bumper-missing",
            )
        finally:
            loguru_logger.remove(handler_id)

        all_text = " ".join(record.message for record in caplog.records)
        assert "req-bumper-missing" in all_text, (
            "Summary must include request_id so operators can filter logs "
            "by a single user submission."
        )
        assert "bumper@0.62:REJECTED_LOW_CONF" in all_text, (
            "The below-threshold bumper must be visible in the diagnostic "
            "log with its actual confidence — otherwise the operator has "
            "no way to tell whether to lower the threshold."
        )
        assert "hood@0.95:ACCEPTED" in all_text


class TestPartsDetectorClassNameValidation:
    def test_warns_on_unknown_classes_but_does_not_raise(self) -> None:
        detector = PartsDetector(model_path="unused")
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: PartType.HOOD.value, 1: "license_plate"},
            results=[],
        )

        detector._assert_model_classes_match_enum()  # must not raise

    def test_does_not_crash_on_model_without_names(self) -> None:
        class _NoNames:
            pass

        detector = PartsDetector(model_path="unused")
        detector._model = _NoNames()  # type: ignore[assignment]
        detector._assert_model_classes_match_enum()
