from dataclasses import dataclass

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


@dataclass(frozen=True)
class PricingRule:
    """Pricing entry for one ``(part_type, damage_type)`` pair.

    Costs are in RUB, durations in hours. The workshop rate card (thesis
    tables 5 and 6) is inherently a range, so we keep four numbers instead
    of a single point estimate. When ``*_min == *_max`` the rule collapses
    to an exact value with zero-width range.
    """

    id: int
    part_type: PartType
    damage_type: DamageType
    labor_hours_min: float
    labor_hours_max: float
    labor_cost_min: float
    labor_cost_max: float
