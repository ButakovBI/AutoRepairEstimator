from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.pricing_rule import PricingRule
from auto_repair_estimator.backend.domain.services.pricing_service import PricingService
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageSource, DamageType, PartType
from auto_repair_estimator.backend.use_cases.calculate_pricing import CalculatePricingInput, CalculatePricingUseCase


class InMemoryDamageRepo:
    def __init__(self, damages: list[DetectedDamage]) -> None:
        self._damages = damages

    async def get_by_request_id(self, request_id: str) -> list[DetectedDamage]:
        return [d for d in self._damages if d.request_id == request_id]

    async def get(self, damage_id: str) -> DetectedDamage | None:
        return next((d for d in self._damages if d.id == damage_id), None)

    async def add(self, damage: DetectedDamage) -> None:
        self._damages.append(damage)

    async def update(self, damage: DetectedDamage) -> None:
        pass

    async def soft_delete(self, damage_id: str) -> None:
        pass


class InMemoryPricingRuleRepo:
    def __init__(self, rules: list[PricingRule]) -> None:
        self._rules = {(r.part_type, r.damage_type): r for r in rules}

    async def get_rule(self, part_type: PartType, damage_type: DamageType) -> PricingRule | None:
        return self._rules.get((part_type, damage_type))

    async def get_all(self) -> list[PricingRule]:
        return list(self._rules.values())


def _damage(part: PartType, damage: DamageType, deleted: bool = False) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-1",
        damage_type=damage,
        part_type=part,
        source=DamageSource.MANUAL,
        is_deleted=deleted,
    )


def _rule(part: PartType, damage: DamageType, hours: float, cost: float) -> PricingRule:
    return PricingRule(id=1, part_type=part, damage_type=damage, labor_hours=hours, labor_cost=cost)


@pytest.mark.anyio
async def test_calculate_pricing_sums_active() -> None:
    damages = [
        _damage(PartType.HOOD, DamageType.SCRATCH),
        _damage(PartType.BUMPER_FRONT, DamageType.DENT),
        _damage(PartType.HOOD, DamageType.RUST, deleted=True),
    ]
    rules = [
        _rule(PartType.HOOD, DamageType.SCRATCH, 1.5, 1200.0),
        _rule(PartType.BUMPER_FRONT, DamageType.DENT, 2.0, 1500.0),
    ]
    use_case = CalculatePricingUseCase(
        damage_repository=InMemoryDamageRepo(damages),
        pricing_service=PricingService(_rule_repository=InMemoryPricingRuleRepo(rules)),
    )
    result = await use_case.execute(CalculatePricingInput(request_id="req-1"))

    assert result.total_cost == pytest.approx(2700.0)
    assert result.total_hours == pytest.approx(3.5)
    assert len(result.breakdown) == 2


@pytest.mark.anyio
async def test_calculate_pricing_no_damages_returns_zero() -> None:
    use_case = CalculatePricingUseCase(
        damage_repository=InMemoryDamageRepo([]),
        pricing_service=PricingService(_rule_repository=InMemoryPricingRuleRepo([])),
    )
    result = await use_case.execute(CalculatePricingInput(request_id="req-1"))

    assert result.total_cost == 0.0
    assert result.total_hours == 0.0
    assert result.breakdown == []
