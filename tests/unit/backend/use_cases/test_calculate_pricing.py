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


@pytest.mark.anyio
async def test_calculate_pricing_sums_active_damages_in_ranges() -> None:
    """Soft-deleted damages must not contribute to either bound of the total."""
    damages = [
        _damage(PartType.HOOD, DamageType.SCRATCH),
        _damage(PartType.BUMPER, DamageType.DENT),
        _damage(PartType.HOOD, DamageType.RUST, deleted=True),
    ]
    # Using thesis-table values so the math stays tied to real requirements:
    # hood × scratch: 10-18k / 8h; bumper × dent: 3-5k / 8-16h.
    rules = [
        _rule(PartType.HOOD, DamageType.SCRATCH, 8, 8, 10_000, 18_000),
        _rule(PartType.BUMPER, DamageType.DENT, 8, 16, 3_000, 5_000),
    ]
    use_case = CalculatePricingUseCase(
        damage_repository=InMemoryDamageRepo(damages),
        pricing_service=PricingService(_rule_repository=InMemoryPricingRuleRepo(rules)),
    )

    result = await use_case.execute(CalculatePricingInput(request_id="req-1"))

    assert result.total_cost_min == pytest.approx(13_000.0)
    assert result.total_cost_max == pytest.approx(23_000.0)
    assert result.total_hours_min == pytest.approx(16.0)
    assert result.total_hours_max == pytest.approx(24.0)
    assert len(result.breakdown) == 2


@pytest.mark.anyio
async def test_calculate_pricing_no_damages_returns_zero_range() -> None:
    use_case = CalculatePricingUseCase(
        damage_repository=InMemoryDamageRepo([]),
        pricing_service=PricingService(_rule_repository=InMemoryPricingRuleRepo([])),
    )

    result = await use_case.execute(CalculatePricingInput(request_id="req-1"))

    assert result.total_cost_min == 0.0
    assert result.total_cost_max == 0.0
    assert result.total_hours_min == 0.0
    assert result.total_hours_max == 0.0
    assert result.breakdown == []
