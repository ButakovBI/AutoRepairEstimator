from typing import Protocol

from auto_repair_estimator.backend.domain.entities.pricing_rule import PricingRule
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


class PricingRuleRepository(Protocol):
    async def get_rule(self, part_type: PartType, damage_type: DamageType) -> PricingRule | None: ...

    async def get_all(self) -> list[PricingRule]: ...
