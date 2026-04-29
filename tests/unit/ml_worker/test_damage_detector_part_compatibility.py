"""Tests for DamageDetector's part↔damage compatibility filter.

Motivation: the user reported a concrete failure mode where the damage
model, running on a door crop, fired ``broken_headlight`` — a physically
impossible combination that then showed up in the "found damages" list
sent to the user as "Дверь — Разбитая фара" and broke pricing. The filter
must drop such cross-part detections at the ML boundary so the bot and
backend never see them.

We reuse the _FakeYOLO harness from ``test_damage_detector`` in spirit
(structural duck-type) without importing it, because these tests target
the filter that sits *between* enum validation and emit — a narrow slice
that benefits from its own focused fixtures.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageType,
    PartType,
)
from auto_repair_estimator.ml_worker.inference.damage_detector import DamageDetector


@dataclass
class _Box:
    conf: list[float]
    cls: list[int]


@dataclass
class _Result:
    boxes: list[_Box]
    names: dict[int, str]
    masks: Any | None = None


class _YOLO:
    def __init__(self, names: dict[int, str], results: list[_Result]) -> None:
        self.names = names
        self._results = results

    def __call__(self, _img: Any, **_kwargs: Any) -> list[_Result]:
        return self._results


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(1, 1, 1)).save(buf, format="JPEG")
    return buf.getvalue()


def _detector_with(classes: dict[int, str], hits: list[tuple[int, float]]) -> DamageDetector:
    detector = DamageDetector(model_path="unused", confidence_threshold=0.1)
    boxes = [_Box(conf=[conf], cls=[cls_id]) for cls_id, conf in hits]
    detector._model = _YOLO(  # type: ignore[attr-defined]
        names=classes,
        results=[_Result(boxes=boxes, names=classes, masks=None)],
    )
    return detector


class TestPartDamageCompatibilityFilter:
    def test_drops_broken_headlight_detected_on_a_door_crop(self) -> None:
        """The exact regression in the user's screenshot."""

        classes = {0: DamageType.BROKEN_HEADLIGHT.value, 1: DamageType.SCRATCH.value}
        detector = _detector_with(classes, hits=[(0, 0.95), (1, 0.9)])

        detections = detector.predict(
            crop_bytes=_jpeg(),
            part_type=PartType.DOOR.value,
            request_id="r",
            crop_index=0,
            bucket="b",
        )

        damage_types = {d.damage_type for d in detections}
        # broken_headlight on a door is physically impossible — must be filtered.
        assert DamageType.BROKEN_HEADLIGHT.value not in damage_types
        # A legitimate body-panel damage on the door still passes through.
        assert DamageType.SCRATCH.value in damage_types

    def test_drops_broken_glass_on_hood(self) -> None:
        """Another cross-part false positive: windows don't live on the hood."""

        classes = {0: DamageType.BROKEN_GLASS.value, 1: DamageType.DENT.value}
        detector = _detector_with(classes, hits=[(0, 0.9), (1, 0.9)])

        detections = detector.predict(
            crop_bytes=_jpeg(),
            part_type=PartType.HOOD.value,
            request_id="r",
            crop_index=0,
            bucket="b",
        )
        damage_types = {d.damage_type for d in detections}
        assert DamageType.BROKEN_GLASS.value not in damage_types
        assert DamageType.DENT.value in damage_types

    def test_keeps_compatible_combinations_unchanged(self) -> None:
        """Windshield + broken_glass is exactly the only allowed pair."""

        classes = {0: DamageType.BROKEN_GLASS.value}
        detector = _detector_with(classes, hits=[(0, 0.9)])

        detections = detector.predict(
            crop_bytes=_jpeg(),
            part_type=PartType.FRONT_WINDSHIELD.value,
            request_id="r",
            crop_index=0,
            bucket="b",
        )
        assert len(detections) == 1
        assert detections[0].damage_type == DamageType.BROKEN_GLASS.value

    def test_unknown_part_type_falls_back_to_no_filter(self) -> None:
        """If the part-type string isn't in PartType (e.g. an older parts
        model emitted a legacy label), the compatibility filter must
        *not* reject everything — the enum filter plus backend validation
        downstream handle that case. Regressing to "drop all" would
        silently erase legitimate damages whenever the parts model is
        ahead of the enum.
        """

        classes = {0: DamageType.SCRATCH.value}
        detector = _detector_with(classes, hits=[(0, 0.9)])

        detections = detector.predict(
            crop_bytes=_jpeg(),
            part_type="mystery_part_from_future_model",
            request_id="r",
            crop_index=0,
            bucket="b",
        )
        assert len(detections) == 1
        assert detections[0].damage_type == DamageType.SCRATCH.value

    def test_propagates_crop_box_pixels_to_detections(self) -> None:
        """The composer relies on ``crop_box_pixels`` being threaded from
        cropper → detector → detection to place masks correctly.
        """

        classes = {0: DamageType.SCRATCH.value}
        detector = _detector_with(classes, hits=[(0, 0.9)])

        detections = detector.predict(
            crop_bytes=_jpeg(),
            part_type=PartType.DOOR.value,
            request_id="r",
            crop_index=0,
            bucket="b",
            crop_box_pixels=(100, 200, 300, 400),
        )
        assert len(detections) == 1
        assert detections[0].crop_box_pixels == (100, 200, 300, 400)
