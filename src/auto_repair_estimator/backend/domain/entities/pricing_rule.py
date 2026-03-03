from dataclasses import dataclass

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


@dataclass(frozen=True)
class PricingRule:
    id: int
    part_type: PartType
    damage_type: DamageType
    labor_hours: float
    labor_cost: float

