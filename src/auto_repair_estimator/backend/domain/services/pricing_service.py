from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.pricing_result import PricingResult
from auto_repair_estimator.backend.domain.interfaces.pricing_rule_repository import PricingRuleRepository


@dataclass
class PricingService:
    _rule_repository: PricingRuleRepository

    async def calculate(self, request_id: str, damages: list[DetectedDamage]) -> PricingResult:
        active = [d for d in damages if not d.is_deleted]
        logger.debug("Calculating pricing for request={} active_damages={}", request_id, len(active))

        total_cost = 0.0
        total_hours = 0.0
        breakdown: list[dict[str, Any]] = []

        for damage in active:
            rule = await self._rule_repository.get_rule(damage.part_type, damage.damage_type)
            if rule is None:
                logger.warning("No pricing rule for part_type={} damage_type={}", damage.part_type, damage.damage_type)
                continue
            total_cost += rule.labor_cost
            total_hours += rule.labor_hours
            breakdown.append(
                {
                    "damage_id": damage.id,
                    "part_type": damage.part_type.value,
                    "damage_type": damage.damage_type.value,
                    "cost": rule.labor_cost,
                    "hours": rule.labor_hours,
                }
            )

        logger.info("Pricing result for request={}: cost={} hours={}", request_id, total_cost, total_hours)
        return PricingResult(request_id=request_id, total_hours=total_hours, total_cost=total_cost, breakdown=breakdown)
