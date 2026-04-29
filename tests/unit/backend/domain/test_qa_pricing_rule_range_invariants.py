"""QA: PricingRule must reject physically impossible ranges.

The rate card encoded in ``PricingRule`` uses four numbers to describe a
``[min, max]`` interval for both cost and hours. Nothing in the current
entity enforces ``min <= max`` — a typo in the SQL seed or a bug in an
admin UI could persist ``labor_cost_min=30_000, labor_cost_max=10_000``,
flipping the range. The ``PricingService`` would then produce totals
with ``total_cost_min > total_cost_max`` and the bot would render
nonsense like ``30,000–10,000 руб.``.

The schema's CHECK constraints (``labor_hours_min <= labor_hours_max``
and ``labor_cost_min_rub <= labor_cost_max_rub``) only protect the DB
layer. The domain entity is the second line of defence and should
refuse inverted ranges at construction time.

These tests encode the invariant contract — they will fail until the
entity enforces it in ``__post_init__``.
"""

from __future__ import annotations

import pytest

from auto_repair_estimator.backend.domain.entities.pricing_rule import PricingRule
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


def _rule(
    *,
    hours_min: float,
    hours_max: float,
    cost_min: float,
    cost_max: float,
) -> PricingRule:
    """Helper that invokes the public constructor; defaults to a valid
    door-dent rule (table 5: 23-30 тыс. руб., 16-24 h)."""
    return PricingRule(
        id=1,
        part_type=PartType.DOOR,
        damage_type=DamageType.DENT,
        labor_hours_min=hours_min,
        labor_hours_max=hours_max,
        labor_cost_min=cost_min,
        labor_cost_max=cost_max,
    )


def test_valid_range_constructs_without_error() -> None:
    """Guard: make sure the helper itself isn't accidentally broken."""
    rule = _rule(hours_min=16, hours_max=24, cost_min=23_000, cost_max=30_000)
    assert rule.labor_hours_min <= rule.labor_hours_max
    assert rule.labor_cost_min <= rule.labor_cost_max


def test_inverted_cost_range_is_rejected_at_construction() -> None:
    """``min > max`` for cost is physically impossible — the entity must
    not allow a caller to build it. Without this check a bad SQL seed
    would propagate straight into the user's estimate."""

    with pytest.raises((ValueError, AssertionError)):
        _rule(hours_min=16, hours_max=24, cost_min=30_000, cost_max=10_000)


def test_inverted_hours_range_is_rejected_at_construction() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _rule(hours_min=24, hours_max=16, cost_min=23_000, cost_max=30_000)


def test_negative_cost_is_rejected() -> None:
    """A negative cost would let a single damage zero out or invert the
    total — the service does plain sum aggregation without a floor."""
    with pytest.raises((ValueError, AssertionError)):
        _rule(hours_min=1, hours_max=1, cost_min=-1, cost_max=0)


def test_negative_hours_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _rule(hours_min=-1, hours_max=0, cost_min=0, cost_max=0)
