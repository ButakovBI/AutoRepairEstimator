from dataclasses import dataclass

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


@dataclass(frozen=True)
class PricingRule:
    """Pricing entry for one ``(part_type, damage_type)`` pair.

    Costs are in RUB, durations in hours. The workshop rate card (thesis
    tables 5 and 6) is inherently a range, so we keep four numbers instead
    of a single point estimate. When ``*_min == *_max`` the rule collapses
    to an exact value with zero-width range.

    Invariants enforced at construction (first line of defence before the
    SQL CHECK constraints in ``pricing_rules``):
      * ``labor_hours_min <= labor_hours_max``
      * ``labor_cost_min <= labor_cost_max``
      * all four numbers are non-negative.
    """

    id: int
    part_type: PartType
    damage_type: DamageType
    labor_hours_min: float
    labor_hours_max: float
    labor_cost_min: float
    labor_cost_max: float

    def __post_init__(self) -> None:
        if self.labor_hours_min < 0 or self.labor_hours_max < 0:
            raise ValueError(
                f"labor_hours must be non-negative; got min={self.labor_hours_min}, "
                f"max={self.labor_hours_max} for {self.part_type.value}+{self.damage_type.value}"
            )
        if self.labor_cost_min < 0 or self.labor_cost_max < 0:
            raise ValueError(
                f"labor_cost must be non-negative; got min={self.labor_cost_min}, "
                f"max={self.labor_cost_max} for {self.part_type.value}+{self.damage_type.value}"
            )
        if self.labor_hours_min > self.labor_hours_max:
            raise ValueError(
                f"labor_hours_min ({self.labor_hours_min}) must be <= labor_hours_max "
                f"({self.labor_hours_max}) for {self.part_type.value}+{self.damage_type.value}"
            )
        if self.labor_cost_min > self.labor_cost_max:
            raise ValueError(
                f"labor_cost_min ({self.labor_cost_min}) must be <= labor_cost_max "
                f"({self.labor_cost_max}) for {self.part_type.value}+{self.damage_type.value}"
            )
