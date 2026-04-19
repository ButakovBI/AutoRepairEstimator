"""Compute a repair estimate from a list of active damages.

The service aggregates ``[min, max]`` cost and duration ranges across all
damages, and collects user-facing notes that don't fit the priced-row model:

* ``flat_tire`` or any ``wheel`` damage -> routed to a tyre shop (no price
  is added; the user gets a textual instruction instead).
* ``scratch`` damages -> appended with the cheaper polishing alternative
  (see ``pricing_constants``) so the user can decide which work to pick.

When a ``(part, damage)`` pair has no matching rule (e.g. ``crack`` on a
glass part) the damage is silently skipped with a warning log — upstream
validation should prevent this path in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.pricing_result import PricingResult
from auto_repair_estimator.backend.domain.interfaces.pricing_rule_repository import PricingRuleRepository
from auto_repair_estimator.backend.domain.value_objects.pricing_constants import (
    POLISH_COST_RUB,
    POLISH_HOURS,
    TYRE_SHOP_NOTE,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


@dataclass
class PricingService:
    _rule_repository: PricingRuleRepository

    async def calculate(self, request_id: str, damages: list[DetectedDamage]) -> PricingResult:
        active = [d for d in damages if not d.is_deleted]
        logger.debug("Calculating pricing for request={} active_damages={}", request_id, len(active))

        total_cost_min = 0.0
        total_cost_max = 0.0
        total_hours_min = 0.0
        total_hours_max = 0.0
        breakdown: list[dict[str, Any]] = []
        notes: list[str] = []
        wheel_reported = False
        scratch_count = 0

        for damage in active:
            # Wheel damage is routed to a tyre shop regardless of damage_type.
            # We don't attach a price row — just a single note per request.
            if damage.part_type is PartType.WHEEL:
                if not wheel_reported:
                    notes.append(TYRE_SHOP_NOTE)
                    wheel_reported = True
                continue

            rule = await self._rule_repository.get_rule(damage.part_type, damage.damage_type)
            if rule is None:
                logger.warning(
                    "No pricing rule for part_type={} damage_type={} — skipping",
                    damage.part_type,
                    damage.damage_type,
                )
                continue

            total_cost_min += rule.labor_cost_min
            total_cost_max += rule.labor_cost_max
            total_hours_min += rule.labor_hours_min
            total_hours_max += rule.labor_hours_max
            breakdown.append(
                {
                    "damage_id": damage.id,
                    "part_type": damage.part_type.value,
                    "damage_type": damage.damage_type.value,
                    "cost_min": rule.labor_cost_min,
                    "cost_max": rule.labor_cost_max,
                    "hours_min": rule.labor_hours_min,
                    "hours_max": rule.labor_hours_max,
                }
            )

            if damage.damage_type is DamageType.SCRATCH:
                scratch_count += 1

        if scratch_count > 0:
            polish_hours = POLISH_HOURS * scratch_count
            polish_cost = POLISH_COST_RUB * scratch_count
            notes.append(
                f"Для царапин в смете учтена покраска (верхний вариант). "
                f"Если царапины неглубокие и достаточно полировки, "
                f"работа по {scratch_count} царапин(ам) займёт около "
                f"{polish_hours:.0f} ч и обойдётся примерно в {polish_cost:,.0f} руб."
            )

        logger.info(
            "Pricing result for request={}: cost=[{}..{}] hours=[{}..{}] notes={}",
            request_id,
            total_cost_min,
            total_cost_max,
            total_hours_min,
            total_hours_max,
            len(notes),
        )
        return PricingResult(
            request_id=request_id,
            total_hours_min=total_hours_min,
            total_hours_max=total_hours_max,
            total_cost_min=total_cost_min,
            total_cost_max=total_cost_max,
            breakdown=breakdown,
            notes=notes,
        )
