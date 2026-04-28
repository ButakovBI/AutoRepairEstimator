"""Guard rails around the ML confidence thresholds SSOT.

These tests don't verify YOLO — they verify that the thresholds the
business cares about live in exactly one place, and that the rest of
the system reads from that place. If someone later copies a magic
0.7 into ``ml_worker/config.py`` or hard-codes a different default in
a detector, these tests must fail.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from auto_repair_estimator.backend.domain.value_objects.ml_thresholds import (
    DAMAGES_CONFIDENCE_BY_CLASS,
    DAMAGES_CONFIDENCE_FLOOR,
    PARTS_CONFIDENCE_THRESHOLD,
    damages_threshold_for,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType
from auto_repair_estimator.ml_worker.config import MLWorkerConfig


class TestPartsThresholdIsBusinessCalibrated:
    def test_parts_threshold_matches_current_policy(self) -> None:
        # 0.5 is the current operational calibration — biased towards
        # catching borderline detections (bumpers on busy backgrounds)
        # and relying on part↔damage compatibility filtering to suppress
        # false positives downstream. If this test fails, update
        # ml_thresholds.py AND re-read its module docstring before
        # adjusting the number.
        assert PARTS_CONFIDENCE_THRESHOLD == pytest.approx(0.5)

    def test_parts_threshold_in_valid_probability_range(self) -> None:
        # YOLO confidences are probabilities in [0, 1]; a threshold
        # outside that range silently disables or saturates filtering.
        assert 0.0 < PARTS_CONFIDENCE_THRESHOLD < 1.0


class TestDamagesPerClassThresholdsAreBusinessCalibrated:
    """Pin the per-class cutoffs so bumps are deliberate code reviews.

    Each assertion documents *why* the class lives in its tier — if
    you change the number here, also update the rationale in
    ``ml_thresholds.py``'s module docstring so the policy stays
    self-explanatory for future maintainers.
    """

    def test_scratch_threshold_is_high_tier(self) -> None:
        # 0.50 — well-represented in training data, model is confident.
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.SCRATCH] == pytest.approx(0.50)

    def test_flat_tire_threshold_is_high_tier(self) -> None:
        # 0.50 — visually distinctive, no need for low recall bar.
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.FLAT_TIRE] == pytest.approx(0.50)

    def test_broken_glass_threshold_is_mid_tier(self) -> None:
        # 0.40 — distinctive (cracks on transparent surface) but rarer
        # in training data than scratches.
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.BROKEN_GLASS] == pytest.approx(0.40)

    def test_broken_headlight_threshold_is_mid_tier(self) -> None:
        # 0.40 — same tier as broken_glass for consistent windowed-defect
        # behaviour across light vs glass surfaces.
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.BROKEN_HEADLIGHT] == pytest.approx(0.40)

    def test_rust_threshold_is_low_tier(self) -> None:
        # 0.30 — rare class, lower bar to keep recall up for early-stage
        # corrosion that's still pricable.
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.RUST] == pytest.approx(0.30)

    def test_paint_chip_threshold_is_low_tier(self) -> None:
        # 0.30 — rare; tier matched with rust per business decision.
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.PAINT_CHIP] == pytest.approx(0.30)

    def test_dent_threshold_is_lowest_tier(self) -> None:
        # 0.25 — weakest model confidence in training (subtle geometry);
        # user still triages before pricing, so a low bar is acceptable.
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.DENT] == pytest.approx(0.25)

    def test_crack_threshold_is_lowest_tier(self) -> None:
        # 0.25 — same tier as dent (rarest geometric defect class).
        assert DAMAGES_CONFIDENCE_BY_CLASS[DamageType.CRACK] == pytest.approx(0.25)


class TestDamagesThresholdsContractInvariants:
    def test_every_damage_class_has_an_explicit_threshold(self) -> None:
        # Exhaustiveness is the *whole point* of using an enum-keyed
        # dict: a missing class would silently fall through to the
        # floor and might disappear from the user's list. The runtime
        # guard in ml_thresholds.py would have already raised at import
        # time, but having a behavioural test makes the contract
        # explicit in the test suite for future code reviews.
        assert set(DAMAGES_CONFIDENCE_BY_CLASS.keys()) == set(DamageType)

    def test_all_thresholds_are_valid_probabilities(self) -> None:
        # Out-of-range values silently disable (≤0) or saturate (≥1)
        # per-class filtering — both are confusing failure modes.
        for cls, value in DAMAGES_CONFIDENCE_BY_CLASS.items():
            assert 0.0 < value < 1.0, f"{cls.value}={value} outside (0,1)"

    def test_floor_is_min_of_per_class_values(self) -> None:
        # Detector uses FLOOR as YOLO ``conf=`` argument; if it diverged
        # from the actual minimum, low-threshold classes would be
        # silently NMS-filtered before our per-class check ran.
        assert DAMAGES_CONFIDENCE_FLOOR == pytest.approx(min(DAMAGES_CONFIDENCE_BY_CLASS.values()))


class TestDamagesThresholdLookupHelper:
    def test_lookup_by_enum_returns_configured_cutoff(self) -> None:
        # The detector calls this helper with the model's class string
        # — but framework-internal callers (tests, future ops tools)
        # should also be able to use the enum directly without
        # round-tripping through ``.value``.
        assert damages_threshold_for(DamageType.SCRATCH) == pytest.approx(0.50)
        assert damages_threshold_for(DamageType.DENT) == pytest.approx(0.25)

    def test_lookup_by_string_returns_configured_cutoff(self) -> None:
        # The hot path: YOLO returns class names as strings. The
        # helper must accept them without coercion at the call site.
        assert damages_threshold_for("crack") == pytest.approx(0.25)
        assert damages_threshold_for("rust") == pytest.approx(0.30)

    def test_unknown_class_falls_back_to_floor(self) -> None:
        # A future model variant might emit a label we haven't wired up
        # to DamageType yet. The detector's enum filter rejects such
        # detections anyway, but having a numeric fallback lets callers
        # compare safely without special-casing None.
        assert damages_threshold_for("aliens_made_this_dent") == pytest.approx(
            DAMAGES_CONFIDENCE_FLOOR
        )


class TestMLWorkerConfigHonoursTheSSOT:
    """MLWorkerConfig must not re-invent its own defaults."""

    def test_parts_default_comes_from_domain_ssot(self) -> None:
        # Wipe ambient env so we observe the code default, not whatever
        # developer's .env happens to define.
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("PARTS_CONFIDENCE_THRESHOLD", "DAMAGES_CONFIDENCE_THRESHOLD"):
                os.environ.pop(key, None)
            config = MLWorkerConfig(_env_file=None)  # type: ignore[call-arg]

        assert config.parts_confidence_threshold == PARTS_CONFIDENCE_THRESHOLD

    def test_damages_default_is_none_so_detector_uses_per_class_ssot(self) -> None:
        # The damages field is intentionally a uniform-override knob,
        # not a default value. None means "let the detector read the
        # per-class policy from ml_thresholds.py" — the only way to
        # keep the per-class SSOT authoritative.
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("PARTS_CONFIDENCE_THRESHOLD", "DAMAGES_CONFIDENCE_THRESHOLD"):
                os.environ.pop(key, None)
            config = MLWorkerConfig(_env_file=None)  # type: ignore[call-arg]

        assert config.damages_confidence_threshold is None

    def test_env_can_uniform_override_for_ops(self) -> None:
        # The uniform override is the ops escape hatch for runtime
        # experiments without rebuilding the ML worker image. It must
        # parse as a float and reach the config field.
        env = {
            "PARTS_CONFIDENCE_THRESHOLD": "0.33",
            "DAMAGES_CONFIDENCE_THRESHOLD": "0.11",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = MLWorkerConfig(_env_file=None)  # type: ignore[call-arg]

        assert config.parts_confidence_threshold == pytest.approx(0.33)
        assert config.damages_confidence_threshold == pytest.approx(0.11)
