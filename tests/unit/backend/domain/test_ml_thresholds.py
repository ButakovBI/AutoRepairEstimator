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
    DAMAGES_CONFIDENCE_THRESHOLD,
    PARTS_CONFIDENCE_THRESHOLD,
)
from auto_repair_estimator.ml_worker.config import MLWorkerConfig


class TestMLThresholdsAreBusinessCalibrated:
    """Pin the numbers so bumps are intentional reviews, not drift."""

    def test_parts_threshold_matches_current_policy(self) -> None:
        # 0.5 is the current operational calibration — biased towards
        # catching borderline detections (bumpers on busy backgrounds)
        # and relying on part↔damage compatibility filtering to suppress
        # false positives downstream. If this test fails, update
        # ml_thresholds.py AND re-read its module docstring before
        # adjusting the number.
        assert PARTS_CONFIDENCE_THRESHOLD == pytest.approx(0.5)

    def test_damages_threshold_matches_current_policy(self) -> None:
        # 0.2 is deliberately low so every likely damage reaches the
        # user-facing "обнаруженные повреждения" list for explicit
        # confirm/reject; a higher cutoff silently hides real damages.
        assert DAMAGES_CONFIDENCE_THRESHOLD == pytest.approx(0.2)

    def test_thresholds_are_in_valid_probability_range(self) -> None:
        # YOLO confidences are probabilities in [0, 1]; a threshold
        # outside that range silently disables or saturates filtering.
        for value in (PARTS_CONFIDENCE_THRESHOLD, DAMAGES_CONFIDENCE_THRESHOLD):
            assert 0.0 < value < 1.0


class TestMLWorkerConfigReadsFromSSOT:
    """MLWorkerConfig must not re-invent its own defaults."""

    def test_defaults_come_from_domain_ssot(self) -> None:
        # Wipe any ambient env so we observe the code default, not
        # whatever developer's .env happens to define.
        with mock.patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            for key in ("PARTS_CONFIDENCE_THRESHOLD", "DAMAGES_CONFIDENCE_THRESHOLD"):
                os.environ.pop(key, None)
            config = MLWorkerConfig(_env_file=None)  # type: ignore[call-arg]

        assert config.parts_confidence_threshold == PARTS_CONFIDENCE_THRESHOLD
        assert config.damages_confidence_threshold == DAMAGES_CONFIDENCE_THRESHOLD

    def test_env_can_still_override_for_ops(self) -> None:
        # The SSOT is the default, not a hard-wired constant: operators
        # must retain the escape hatch for runtime experiments without
        # rebuilding the ML worker image.
        env = {
            "PARTS_CONFIDENCE_THRESHOLD": "0.33",
            "DAMAGES_CONFIDENCE_THRESHOLD": "0.11",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = MLWorkerConfig(_env_file=None)  # type: ignore[call-arg]

        assert config.parts_confidence_threshold == pytest.approx(0.33)
        assert config.damages_confidence_threshold == pytest.approx(0.11)
