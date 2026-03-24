from __future__ import annotations

import asyncpg

from auto_repair_estimator.backend.domain.entities.pricing_rule import PricingRule
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


class PostgresPricingRuleRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_rule(self, part_type: PartType, damage_type: DamageType) -> PricingRule | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM pricing_rules WHERE part_type=$1 AND damage_type=$2",
            part_type.value,
            damage_type.value,
        )
        if row is None:
            return None
        return self._from_row(row)

    async def get_all(self) -> list[PricingRule]:
        rows = await self._pool.fetch("SELECT * FROM pricing_rules ORDER BY id")
        return [self._from_row(r) for r in rows]

    @staticmethod
    def _from_row(row: asyncpg.Record) -> PricingRule:
        return PricingRule(
            id=row["id"],
            part_type=PartType(row["part_type"]),
            damage_type=DamageType(row["damage_type"]),
            labor_hours=row["labor_hours"],
            labor_cost=row["labor_cost_rub"],
        )
