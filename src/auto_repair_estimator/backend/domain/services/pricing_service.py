"""Compute a repair estimate from a list of active damages.

The service aggregates ``[min, max]`` cost and duration ranges across
the damages that end up on the priced path. Before it looks up any
pricing rule, it runs :func:`aggregate_damages_for_pricing` to apply
two business rules that cannot be expressed in the rate card alone:

* **Per-part dedup.** Two scratches on the same door are priced as one
  painting — the door is painted once regardless of the number of
  scratches. The aggregator collapses duplicates per physical part
  (or per ``part_type`` for manual-mode damages where no part
  instance id exists).
* **Replacement supersession.** If a part has a replacement-class
  damage (paint_chip / crack / broken_glass / broken_headlight), the
  whole part is priced as replacement and any non-replacement damages
  on that part drop out — you don't paint a panel you're replacing.

Additional behaviours preserved from the earlier version:

* Wheel damages are routed to a single tyre-shop note, not a priced
  row (``TYRE_SHOP_NOTE``).
* Scratches emit a user-visible "polishing alternative" note, now
  scoped to *parts with scratch* (after dedup) rather than raw
  scratch count — matching the new "paint once per part" contract.
* Unknown ``(part, damage)`` combos (no pricing rule) are surfaced as
  user notes so the bot can explain why they didn't appear in the
  total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.pricing_result import PricingResult
from auto_repair_estimator.backend.domain.interfaces.pricing_rule_repository import PricingRuleRepository
from auto_repair_estimator.backend.domain.services.damage_aggregator import (
    aggregate_damages_for_pricing,
)
from auto_repair_estimator.backend.domain.value_objects.damage_severity import (
    causes_replacement,
)
from auto_repair_estimator.backend.domain.value_objects.labels import DAMAGE_LABELS, PART_LABELS
from auto_repair_estimator.backend.domain.value_objects.pricing_constants import (
    POLISH_COST_RUB,
    POLISH_HOURS,
    TYRE_SHOP_NOTE,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType

# Breakdown row treatment markers. These surface in the API response so
# the bot can render a "— замена" suffix without reimplementing the rule
# that decides which damages imply a full-part replacement. Kept as
# public string constants so integration-level test doubles can match
# them without importing the enum.
TREATMENT_REPLACEMENT = "replacement"
TREATMENT_DEFAULT = "default"


@dataclass
class PricingService:
    _rule_repository: PricingRuleRepository

    async def calculate(self, request_id: str, damages: list[DetectedDamage]) -> PricingResult:
        active = [d for d in damages if not d.is_deleted]
        logger.debug(
            "Calculating pricing for request={} active_damages={}",
            request_id,
            len(active),
        )

        # Wheels are routed to the tyre shop out of band — detect first
        # so we can set the note exactly once, regardless of how many
        # wheel damages the aggregator would otherwise collapse.
        wheel_damages = [d for d in active if d.part_type is PartType.WHEEL]
        priceable_input = [d for d in active if d.part_type is not PartType.WHEEL]

        aggregation = aggregate_damages_for_pricing(priceable_input)
        if aggregation.dropped_duplicates or aggregation.dropped_by_replacement_supersession:
            logger.info(
                "Pricing aggregation for request={}: duplicates_dropped={} replacement_superseded={}",
                request_id,
                aggregation.dropped_duplicates,
                aggregation.dropped_by_replacement_supersession,
            )

        total_cost_min = 0.0
        total_cost_max = 0.0
        total_hours_min = 0.0
        total_hours_max = 0.0
        breakdown: list[dict[str, Any]] = []
        notes: list[str] = []
        scratch_part_count = 0

        for damage in aggregation.kept:
            rule = await self._rule_repository.get_rule(damage.part_type, damage.damage_type)
            if rule is None:
                # No pricing rule: surface a user-visible note so the
                # bot can explain the omission. See
                # ``test_qa_pricing_service_notes_unpriced_damages``.
                logger.warning(
                    "No pricing rule for part_type={} damage_type={} — adding user note",
                    damage.part_type,
                    damage.damage_type,
                )
                part_label = PART_LABELS.get(damage.part_type.value, damage.part_type.value)
                damage_label = DAMAGE_LABELS.get(damage.damage_type.value, damage.damage_type.value)
                notes.append(
                    f"Комбинация «{part_label} — {damage_label}» не оценивается "
                    f"в рамках кузовного ремонта. Уточните детали у механика "
                    f"или удалите это повреждение из заявки."
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
                    # Treatment is what the bot uses to decide whether to
                    # render a clarifying "— замена" suffix. We key off
                    # the domain predicate rather than hardcoding a set
                    # here so a new replacement-class damage type gets
                    # the label for free via ``damage_severity.py``.
                    "treatment": (
                        TREATMENT_REPLACEMENT if causes_replacement(damage.damage_type) else TREATMENT_DEFAULT
                    ),
                }
            )

            if damage.damage_type is DamageType.SCRATCH:
                # After aggregation, each scratch rep corresponds to one
                # distinct painted part, so the "polishing alternative"
                # note now scales per scratched part. Semantically this
                # matches physical reality: you'd polish each affected
                # panel once.
                scratch_part_count += 1

        if wheel_damages:
            notes.append(TYRE_SHOP_NOTE)

        if scratch_part_count > 0:
            polish_hours = POLISH_HOURS * scratch_part_count
            polish_cost = POLISH_COST_RUB * scratch_part_count
            notes.append(
                f"Для царапин в смете учтена покраска (верхний вариант). "
                f"Если царапины неглубокие и достаточно полировки, "
                f"работа по {scratch_part_count} царапин(ам) займёт около "
                f"{polish_hours:.0f} ч и обойдётся примерно в {polish_cost:,.0f} руб."
            )

        logger.info(
            "Pricing result for request={}: cost=[{}..{}] hours=[{}..{}] notes={} priced_rows={}",
            request_id,
            total_cost_min,
            total_cost_max,
            total_hours_min,
            total_hours_max,
            len(notes),
            len(breakdown),
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
