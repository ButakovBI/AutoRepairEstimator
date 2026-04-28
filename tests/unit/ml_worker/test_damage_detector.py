"""Unit tests for :class:`DamageDetector`.

The real model (``ultralytics.YOLO``) is heavy and pulls in torch. We don't
need it to verify the **contract** the detector exposes to the ML pipeline:

* Unknown class names returned by the model MUST be dropped silently rather
  than propagated downstream, because the backend only accepts values that
  exist in :class:`DamageType`.
* Detections below ``confidence_threshold`` MUST be dropped.
* ``load()`` MUST log class-mismatch warnings instead of raising, so a newer
  model can always be hot-swapped without a worker restart loop.

We substitute ``self._model`` with a hand-rolled fake that implements the
tiny slice of the YOLO interface the detector actually touches: ``__call__``
returns a list of ``Result`` objects with ``.boxes``, ``.masks``, ``.names``.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from PIL import Image

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType
from auto_repair_estimator.ml_worker.inference.damage_detector import (
    DamageDetector,
)


# ---------------------------------------------------------------------------
# Fake YOLO: a minimal structural double of the ultralytics API surface that
# `DamageDetector.predict` and `_assert_model_classes_match_enum` actually use.
# Keeping it in this file (not a fixtures module) makes each test easy to read
# in isolation.
# ---------------------------------------------------------------------------
@dataclass
class _FakeBox:
    # xywhn is unused by the damage detector (only the parts detector reads it),
    # so we only need `conf` and `cls`, both of which YOLO exposes as length-1
    # arrays (simulating tensors). We use lists so tests don't need torch.
    conf: list[float]
    cls: list[int]


class _FakeMasks:
    def __init__(self, mask_arrays: list[np.ndarray]) -> None:
        # Emulate `.data[i].cpu().numpy()` — YOLO returns torch tensors, we
        # return numpy arrays with a `.cpu().numpy()` passthrough shim.
        self.data = [_FakeTensor(m) for m in mask_arrays]


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


@dataclass
class _FakeResult:
    boxes: list[_FakeBox]
    names: dict[int, str]
    masks: _FakeMasks | None = None


class _FakeYOLO:
    """Drop-in for ``ultralytics.YOLO`` — only the fields touched by our code.

    ``__call__`` accepts arbitrary kwargs (``verbose``, ``conf``, ...) so
    the fixture stays compatible when the detector starts passing new
    inference knobs to the underlying model.
    """

    def __init__(self, names: dict[int, str], results: list[_FakeResult]) -> None:
        self.names = names
        self._results = results

    def __call__(self, _image: Any, **_kwargs: Any) -> list[_FakeResult]:
        return self._results


def _jpeg_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    # The predict() implementation opens the bytes with PIL before handing
    # them off to YOLO, so the input must be a decodable JPEG. Size doesn't
    # matter for these contract tests — the model is mocked.
    img = Image.new("RGB", size, color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestDamageDetectorPredictFiltering:
    def test_drops_detections_with_unknown_class_name(self) -> None:
        # Model returns two high-confidence detections: one legit, one rogue
        # class ("alien_damage") that is not in DamageType. The rogue one must
        # be silently dropped so the backend never sees it.
        detector = DamageDetector(model_path="unused", confidence_threshold=0.5)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: DamageType.SCRATCH.value, 1: "alien_damage"},
            results=[
                _FakeResult(
                    boxes=[
                        _FakeBox(conf=[0.9], cls=[0]),  # legit scratch
                        _FakeBox(conf=[0.95], cls=[1]),  # unknown class — must drop
                    ],
                    names={0: DamageType.SCRATCH.value, 1: "alien_damage"},
                    masks=None,
                )
            ],
        )

        detections = detector.predict(
            crop_bytes=_jpeg_bytes(),
            part_type="hood",
            request_id="req-1",
            crop_index=0,
            bucket="crops",
        )

        assert len(detections) == 1
        assert detections[0].damage_type == DamageType.SCRATCH.value

    def test_drops_detections_below_confidence_threshold(self) -> None:
        # Threshold 0.7 — a 0.6 detection must be rejected even if the class
        # is valid. This guards against silent drift if the threshold is ever
        # re-ordered relative to the class-filter branch.
        detector = DamageDetector(model_path="unused", confidence_threshold=0.7)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: DamageType.DENT.value},
            results=[
                _FakeResult(
                    boxes=[_FakeBox(conf=[0.6], cls=[0])],
                    names={0: DamageType.DENT.value},
                    masks=None,
                )
            ],
        )

        detections = detector.predict(
            crop_bytes=_jpeg_bytes(),
            part_type="door",
            request_id="req-1",
            crop_index=0,
            bucket="crops",
        )

        assert detections == []

    def test_raises_when_not_loaded(self) -> None:
        detector = DamageDetector(model_path="unused")
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict(
                crop_bytes=_jpeg_bytes(),
                part_type="hood",
                request_id="r",
                crop_index=0,
                bucket="crops",
            )

    def test_emits_mask_key_only_when_mask_present(self) -> None:
        # mask_key naming is part of the contract with the composer/S3
        # publisher; if a detection has no mask (detection model variant or
        # output truncation) the key must be None so nothing tries to upload
        # a non-existent PNG.
        detector = DamageDetector(model_path="unused", confidence_threshold=0.1)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: DamageType.RUST.value},
            results=[
                _FakeResult(
                    boxes=[_FakeBox(conf=[0.9], cls=[0])],
                    names={0: DamageType.RUST.value},
                    masks=None,
                )
            ],
        )

        detections = detector.predict(
            crop_bytes=_jpeg_bytes(),
            part_type="hood",
            request_id="req-9",
            crop_index=2,
            bucket="crops",
        )

        assert len(detections) == 1
        assert detections[0].mask_key is None


class TestDamageDetectorPerClassThresholds:
    """The detector must apply DIFFERENT cutoffs to different classes.

    Uniform global thresholding silently rejected legitimate dents and
    cracks (which the user complained were "missing") because they
    score lower confidences than the well-represented scratch class.
    These tests pin the per-class behaviour so a future regression
    that flattens thresholds back to a single float is caught.
    """

    def test_low_threshold_class_passes_when_high_threshold_class_would_be_dropped(self) -> None:
        # Two boxes at confidence 0.30: a DENT (cutoff 0.25 → accept)
        # and a SCRATCH (cutoff 0.50 → reject). A legacy uniform
        # threshold could only do one of these decisions, never both.
        thresholds = {DamageType.DENT.value: 0.25, DamageType.SCRATCH.value: 0.50}
        detector = DamageDetector(model_path="unused", thresholds=thresholds)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: DamageType.DENT.value, 1: DamageType.SCRATCH.value},
            results=[
                _FakeResult(
                    boxes=[
                        _FakeBox(conf=[0.30], cls=[0]),  # dent — passes 0.25 cutoff
                        _FakeBox(conf=[0.30], cls=[1]),  # scratch — fails 0.50 cutoff
                    ],
                    names={0: DamageType.DENT.value, 1: DamageType.SCRATCH.value},
                    masks=None,
                )
            ],
        )

        detections = detector.predict(
            crop_bytes=_jpeg_bytes(),
            part_type="door",
            request_id="r",
            crop_index=0,
            bucket="b",
        )

        kinds = {d.damage_type for d in detections}
        assert kinds == {DamageType.DENT.value}, (
            "Per-class thresholds must keep the dent (above its 0.25 cutoff) "
            "and drop the scratch (below its 0.50 cutoff)."
        )

    def test_unknown_class_uses_floor_then_enum_filter_drops_it(self) -> None:
        # An unknown class lacks a per-class entry, so it gets the floor
        # treatment: it survives the conf check at conf >= floor, then is
        # filtered by the DamageType enum check. This guards against a
        # subtle bug where an unknown class with very low conf could be
        # kept as "valid" before being stripped — instead, the verdict
        # should always be REJECTED_UNKNOWN_CLASS.
        thresholds = {DamageType.SCRATCH.value: 0.50}
        detector = DamageDetector(model_path="unused", thresholds=thresholds)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: "future_class_not_in_enum"},
            results=[
                _FakeResult(
                    boxes=[_FakeBox(conf=[0.99], cls=[0])],
                    names={0: "future_class_not_in_enum"},
                    masks=None,
                )
            ],
        )

        detections = detector.predict(
            crop_bytes=_jpeg_bytes(),
            part_type="door",
            request_id="r",
            crop_index=0,
            bucket="b",
        )
        # Enum filter rejects unknown labels regardless of confidence.
        assert detections == []

    def test_uniform_override_via_legacy_kwarg_applies_to_all_classes(self) -> None:
        # Backward compatibility: callers that pass a single
        # ``confidence_threshold`` (env-driven uniform override path,
        # plus older unit tests) must still get uniform behaviour.
        detector = DamageDetector(model_path="unused", confidence_threshold=0.7)
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            names={0: DamageType.DENT.value, 1: DamageType.SCRATCH.value},
            results=[
                _FakeResult(
                    boxes=[
                        _FakeBox(conf=[0.65], cls=[0]),  # dent — would normally pass 0.25
                        _FakeBox(conf=[0.75], cls=[1]),  # scratch — passes uniform 0.7
                    ],
                    names={0: DamageType.DENT.value, 1: DamageType.SCRATCH.value},
                    masks=None,
                )
            ],
        )

        detections = detector.predict(
            crop_bytes=_jpeg_bytes(),
            part_type="door",
            request_id="r",
            crop_index=0,
            bucket="b",
        )
        kinds = {d.damage_type for d in detections}
        # 0.65 < 0.7 uniform → dent dropped; 0.75 > 0.7 → scratch kept.
        assert kinds == {DamageType.SCRATCH.value}

    def test_default_thresholds_come_from_ssot(self) -> None:
        # No kwargs → detector reads DAMAGES_CONFIDENCE_BY_CLASS. This
        # is the contract that prevents drift between the file at the
        # SSOT location and the runtime behaviour.
        from auto_repair_estimator.backend.domain.value_objects.ml_thresholds import (
            DAMAGES_CONFIDENCE_BY_CLASS,
        )

        detector = DamageDetector(model_path="unused")
        for damage_type, cutoff in DAMAGES_CONFIDENCE_BY_CLASS.items():
            assert detector._thresholds[damage_type.value] == pytest.approx(cutoff)


class TestDamageDetectorClassNameValidation:
    def test_warns_on_unknown_classes_but_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        # Mismatch is a deployment smell, not a crash condition — a newer
        # model may legitimately include extra classes we haven't wired up
        # to DamageType yet. Validation must warn, not abort startup.
        detector = DamageDetector(model_path="unused")
        detector._model = _FakeYOLO(  # type: ignore[attr-defined]
            # Mix of supported + one unsupported class name:
            names={0: DamageType.SCRATCH.value, 1: "some_new_damage"},
            results=[],
        )

        # Should not raise even though `some_new_damage` is unknown.
        detector._assert_model_classes_match_enum()

    def test_does_not_crash_when_model_has_no_names_attribute(self) -> None:
        # Some ultralytics versions lazy-populate `.names`. The validator
        # must degrade to a warning — not crash the worker — if it's absent.
        class _NoNames:
            pass

        detector = DamageDetector(model_path="unused")
        detector._model = _NoNames()  # type: ignore[assignment]
        detector._assert_model_classes_match_enum()  # must not raise
