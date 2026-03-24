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


def _rule(part: PartType, damage: DamageType, hours: float, cost: float) -> PricingRule:
    return PricingRule(id=1, part_type=part, damage_type=damage, labor_hours=hours, labor_cost=cost)


def _damage(part: PartType, damage: DamageType, deleted: bool = False) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-1",
        damage_type=damage,
        part_type=part,
        source=DamageSource.MANUAL,
        is_deleted=deleted,
    )


@pytest.mark.anyio
async def test_pricing_sums_active_damages() -> None:
    repo = InMemoryPricingRuleRepository(
        [
            _rule(PartType.HOOD, DamageType.SCRATCH, 1.5, 1200.0),
            _rule(PartType.BUMPER_FRONT, DamageType.DENT, 2.0, 1500.0),
        ]
    )
    service = PricingService(_rule_repository=repo)
    damages = [
        _damage(PartType.HOOD, DamageType.SCRATCH),
        _damage(PartType.BUMPER_FRONT, DamageType.DENT),
    ]
    result = await service.calculate("req-1", damages)

    assert result.total_cost == pytest.approx(2700.0)
    assert result.total_hours == pytest.approx(3.5)
    assert len(result.breakdown) == 2


@pytest.mark.anyio
async def test_pricing_skips_deleted_damages() -> None:
    repo = InMemoryPricingRuleRepository([_rule(PartType.HOOD, DamageType.SCRATCH, 1.5, 1200.0)])
    service = PricingService(_rule_repository=repo)
    damages = [
        _damage(PartType.HOOD, DamageType.SCRATCH),
        _damage(PartType.HOOD, DamageType.SCRATCH, deleted=True),
    ]
    result = await service.calculate("req-1", damages)

    assert result.total_cost == pytest.approx(1200.0)
    assert result.total_hours == pytest.approx(1.5)
    assert len(result.breakdown) == 1


@pytest.mark.anyio
async def test_pricing_empty_damages_returns_zero() -> None:
    repo = InMemoryPricingRuleRepository([])
    service = PricingService(_rule_repository=repo)
    result = await service.calculate("req-1", [])

    assert result.total_cost == 0.0
    assert result.total_hours == 0.0
    assert result.breakdown == []


@pytest.mark.anyio
async def test_pricing_unknown_combination_is_skipped() -> None:
    repo = InMemoryPricingRuleRepository([])
    service = PricingService(_rule_repository=repo)
    damages = [_damage(PartType.HOOD, DamageType.SCRATCH)]
    result = await service.calculate("req-1", damages)

    assert result.total_cost == 0.0
    assert result.total_hours == 0.0
    assert result.breakdown == []


@pytest.mark.anyio
async def test_pricing_breakdown_contains_correct_fields() -> None:
    repo = InMemoryPricingRuleRepository([_rule(PartType.TRUNK, DamageType.RUST, 3.5, 3000.0)])
    service = PricingService(_rule_repository=repo)
    damages = [_damage(PartType.TRUNK, DamageType.RUST)]
    result = await service.calculate("req-1", damages)

    assert len(result.breakdown) == 1
    item = result.breakdown[0]
    assert item["part_type"] == "trunk"
    assert item["damage_type"] == "rust"
    assert item["cost"] == pytest.approx(3000.0)
    assert item["hours"] == pytest.approx(3.5)
