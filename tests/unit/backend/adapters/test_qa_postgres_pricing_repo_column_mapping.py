"""QA: PostgresPricingRuleRepository must read the new min/max columns.

This is a white-box regression test for the DB adapter. The schema in
``docker/init.sql`` was recently migrated from

    labor_hours FLOAT, labor_cost_rub FLOAT

to

    labor_hours_min FLOAT, labor_hours_max FLOAT,
    labor_cost_min_rub FLOAT, labor_cost_max_rub FLOAT

If the repository is ever reverted or someone copy-pastes the old column
names into a new adapter, the pricing result will silently fall back to
``KeyError`` on row lookup or (worse) return incorrect numbers. The
asyncpg ``Record`` object is dict-like, so a stub is enough to exercise
the mapping without a live Postgres.

We don't test the SQL itself (that's integration territory); we test the
Python-level mapping contract that the rest of the code depends on.
"""

from __future__ import annotations

from auto_repair_estimator.backend.adapters.repositories.postgres_pricing_rule_repository import (
    PostgresPricingRuleRepository,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


class _FakeRow(dict):
    """Minimal asyncpg.Record stand-in: dict-like lookup only."""

    # Rows returned by asyncpg support only ``row[key]`` access in our code.


def test_from_row_reads_new_min_max_columns() -> None:
    row = _FakeRow(
        id=7,
        part_type="door",
        damage_type="dent",
        # Values chosen to match thesis table 5 row "Дверь × вмятина":
        # 23-30 тыс. руб., 16-24 h.
        labor_hours_min=16.0,
        labor_hours_max=24.0,
        labor_cost_min_rub=23_000.0,
        labor_cost_max_rub=30_000.0,
    )

    rule = PostgresPricingRuleRepository._from_row(row)

    assert rule.id == 7
    assert rule.part_type is PartType.DOOR
    assert rule.damage_type is DamageType.DENT
    assert rule.labor_hours_min == 16.0
    assert rule.labor_hours_max == 24.0
    assert rule.labor_cost_min == 23_000.0
    assert rule.labor_cost_max == 30_000.0


def test_from_row_does_not_swap_min_and_max() -> None:
    """The column names are long and easy to confuse — verify the adapter
    didn't accidentally cross-wire ``min`` to ``max``."""

    row = _FakeRow(
        id=1,
        part_type="bumper",
        damage_type="dent",
        # Table 5 row "Бампер × вмятина": 3-5 тыс. руб., 1-2 days (8-16 h).
        # Intentionally using non-equal min/max so a swap would flip sign of
        # (labor_hours_max - labor_hours_min) assertions below.
        labor_hours_min=8.0,
        labor_hours_max=16.0,
        labor_cost_min_rub=3_000.0,
        labor_cost_max_rub=5_000.0,
    )

    rule = PostgresPricingRuleRepository._from_row(row)

    assert rule.labor_hours_min < rule.labor_hours_max
    assert rule.labor_cost_min < rule.labor_cost_max


def test_from_row_fails_loud_on_old_column_names() -> None:
    """If someone resurrects the old schema (``labor_cost_rub``) the adapter
    must break visibly — a silent fallback to ``0.0`` would under-charge
    every request. We assert by omission: the old keys alone are insufficient."""

    legacy_row = _FakeRow(
        id=1,
        part_type="door",
        damage_type="dent",
        # Only the legacy columns:
        labor_hours=1.5,
        labor_cost_rub=3_000.0,
    )

    try:
        PostgresPricingRuleRepository._from_row(legacy_row)
    except KeyError:
        return  # Expected: adapter demands the new column names.
    except Exception as exc:  # pragma: no cover - any hard fail is acceptable
        raise AssertionError(
            f"Adapter failed on legacy row but with unexpected exception type: {exc!r}"
        ) from exc
    raise AssertionError(
        "Adapter accepted a row that is missing the new min/max columns. That "
        "would mean the mapping silently picked defaults and produced zeroed "
        "pricing rules."
    )
