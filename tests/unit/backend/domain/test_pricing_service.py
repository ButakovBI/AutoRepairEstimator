from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.pricing_rule import PricingRule
from auto_repair_estimator.backend.domain.services.pricing_service import PricingService
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageSource, DamageType, PartType


class InMemoryPricingRuleRepository:
    def __init__(self, rules: list[PricingRule]) -> None:
        self._rules = {(r.part_type, r.damage_type): r for r in rules}

    async def get_rule(self, part_type: PartType, damage_type: DamageType) -> PricingRule | None:
        return self._rules.get((part_type, damage_type))

    async def get_all(self) -> list[PricingRule]:
        return list(self._rules.values())


def _rule(
    part: PartType,
    damage: DamageType,
    hours_min: float,
    hours_max: float,
    cost_min: float,
    cost_max: float,
) -> PricingRule:
    return PricingRule(
        id=1,
        part_type=part,
        damage_type=damage,
        labor_hours_min=hours_min,
        labor_hours_max=hours_max,
        labor_cost_min=cost_min,
        labor_cost_max=cost_max,
    )


def _damage(part: PartType, damage: DamageType, deleted: bool = False) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-1",
        damage_type=damage,
        part_type=part,
        source=DamageSource.MANUAL,
        is_deleted=deleted,
    )


# Table-5/6 upper bound for Hood × Dent is 30-35 tys. RUB / 2 days (16 h).
# Using real thesis numbers instead of synthetic ones keeps the test meaningful
# even after schema migrations.
_HOOD_DENT_RULE = _rule(PartType.HOOD, DamageType.DENT, 16, 16, 30_000, 35_000)
# Bumper × Dent: 3-5 tys. RUB, 1-2 days (8-16 h)
_BUMPER_DENT_RULE = _rule(PartType.BUMPER, DamageType.DENT, 8, 16, 3_000, 5_000)


@pytest.mark.anyio
async def test_pricing_sums_active_damages_min_and_max_separately() -> None:
    """Two non-overlapping rules: totals must be element-wise sums."""
    repo = InMemoryPricingRuleRepository([_HOOD_DENT_RULE, _BUMPER_DENT_RULE])
    service = PricingService(_rule_repository=repo)
    damages = [
        _damage(PartType.HOOD, DamageType.DENT),
        _damage(PartType.BUMPER, DamageType.DENT),
    ]

    result = await service.calculate("req-1", damages)

    # cost: 30_000 + 3_000 .. 35_000 + 5_000
    assert result.total_cost_min == pytest.approx(33_000.0)
    assert result.total_cost_max == pytest.approx(40_000.0)
    # hours: 16 + 8 .. 16 + 16
    assert result.total_hours_min == pytest.approx(24.0)
    assert result.total_hours_max == pytest.approx(32.0)
    assert len(result.breakdown) == 2


@pytest.mark.anyio
async def test_pricing_skips_deleted_damages() -> None:
    repo = InMemoryPricingRuleRepository([_HOOD_DENT_RULE])
    service = PricingService(_rule_repository=repo)
    damages = [
        _damage(PartType.HOOD, DamageType.DENT),
        _damage(PartType.HOOD, DamageType.DENT, deleted=True),
    ]

    result = await service.calculate("req-1", damages)

    # Only the live damage contributes — otherwise we'd double-charge 60-70k.
    assert result.total_cost_min == pytest.approx(30_000.0)
    assert result.total_cost_max == pytest.approx(35_000.0)
    assert len(result.breakdown) == 1


@pytest.mark.anyio
async def test_pricing_empty_damages_returns_zero_range() -> None:
    repo = InMemoryPricingRuleRepository([])
    service = PricingService(_rule_repository=repo)

    result = await service.calculate("req-1", [])

    assert result.total_cost_min == 0.0
    assert result.total_cost_max == 0.0
    assert result.total_hours_min == 0.0
    assert result.total_hours_max == 0.0
    assert result.breakdown == []
    assert result.notes == []


@pytest.mark.anyio
async def test_pricing_unknown_combination_is_skipped_silently() -> None:
    """``(roof, broken_glass)`` is not in the rate card — service must skip it
    rather than raise, so a partially unknown set of damages still produces
    an estimate for the known ones."""
    repo = InMemoryPricingRuleRepository([])
    service = PricingService(_rule_repository=repo)
    damages = [_damage(PartType.ROOF, DamageType.BROKEN_GLASS)]

    result = await service.calculate("req-1", damages)

    assert result.total_cost_min == 0.0
    assert result.total_cost_max == 0.0
    assert result.breakdown == []


@pytest.mark.anyio
async def test_breakdown_item_exposes_min_max_per_phase() -> None:
    repo = InMemoryPricingRuleRepository([_BUMPER_DENT_RULE])
    service = PricingService(_rule_repository=repo)
    damages = [_damage(PartType.BUMPER, DamageType.DENT)]

    result = await service.calculate("req-1", damages)

    assert len(result.breakdown) == 1
    item = result.breakdown[0]
    assert item["part_type"] == "bumper"
    assert item["damage_type"] == "dent"
    assert item["cost_min"] == pytest.approx(3_000.0)
    assert item["cost_max"] == pytest.approx(5_000.0)
    assert item["hours_min"] == pytest.approx(8.0)
    assert item["hours_max"] == pytest.approx(16.0)


@pytest.mark.anyio
async def test_wheel_damage_emits_tyre_shop_note_and_no_price() -> None:
    """Wheel damage is explicitly out of the body shop's scope per §5 of the
    requirements: no price row, just a routing hint."""
    repo = InMemoryPricingRuleRepository([_BUMPER_DENT_RULE])
    service = PricingService(_rule_repository=repo)
    damages = [
        _damage(PartType.WHEEL, DamageType.FLAT_TIRE),
        _damage(PartType.BUMPER, DamageType.DENT),
    ]

    result = await service.calculate("req-1", damages)

    # Only the bumper damage produced a price row.
    assert len(result.breakdown) == 1
    assert result.breakdown[0]["part_type"] == "bumper"
    # Exactly one tyre-shop note regardless of how many wheel damages appear.
    assert any("шиномонтаж" in n.lower() for n in result.notes)


@pytest.mark.anyio
async def test_multiple_wheel_damages_produce_single_tyre_shop_note() -> None:
    repo = InMemoryPricingRuleRepository([])
    service = PricingService(_rule_repository=repo)
    damages = [
        _damage(PartType.WHEEL, DamageType.FLAT_TIRE),
        _damage(PartType.WHEEL, DamageType.DENT),
    ]

    result = await service.calculate("req-1", damages)

    tyre_notes = [n for n in result.notes if "шиномонтаж" in n.lower()]
    assert len(tyre_notes) == 1
    assert result.breakdown == []


@pytest.mark.anyio
async def test_scratch_adds_polish_note_with_per_scratched_part_accounting() -> None:
    """Two distinct parts each with a scratch → one polish line per part.

    Under the aggregated-pricing contract, multiple scratches on the
    *same* part collapse to one (a door is painted once regardless of
    how many scratches it carries). The polish-note N therefore counts
    scratched *parts*, not raw scratch detections. Two different parts
    with scratches → N=2.
    """

    door_scratch = _rule(PartType.DOOR, DamageType.SCRATCH, 8, 8, 10_000, 18_000)
    hood_scratch = _rule(PartType.HOOD, DamageType.SCRATCH, 8, 8, 10_000, 18_000)
    repo = InMemoryPricingRuleRepository([door_scratch, hood_scratch])
    service = PricingService(_rule_repository=repo)
    damages = [
        _damage(PartType.DOOR, DamageType.SCRATCH),
        _damage(PartType.HOOD, DamageType.SCRATCH),
    ]

    result = await service.calculate("req-1", damages)

    # Two scratched parts × painting rate: 2 * (10k..18k) = 20k..36k.
    assert result.total_cost_min == pytest.approx(20_000.0)
    assert result.total_cost_max == pytest.approx(36_000.0)
    polish_notes = [n for n in result.notes if "полировк" in n.lower()]
    assert len(polish_notes) == 1
    # Polish is 1000 RUB * 2 scratched parts = 2000. "2" must appear in the text.
    assert "2" in polish_notes[0]
