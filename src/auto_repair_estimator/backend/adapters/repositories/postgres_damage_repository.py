from __future__ import annotations

import asyncpg
from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageSource, DamageType, PartType


class PostgresDamageRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add(self, damage: DetectedDamage) -> None:
        logger.debug("INSERT detected_damage id={} request_id={}", damage.id, damage.request_id)
        await self._pool.execute(
            """
            INSERT INTO detected_damages
                (id, request_id, part_id, damage_type, part_type,
                 source, confidence, mask_image_key, is_deleted)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            damage.id,
            damage.request_id,
            damage.part_id,
            damage.damage_type.value,
            damage.part_type.value,
            damage.source.value,
            damage.confidence,
            damage.mask_image_key,
            damage.is_deleted,
        )

    async def get_by_request_id(self, request_id: str) -> list[DetectedDamage]:
        rows = await self._pool.fetch("SELECT * FROM detected_damages WHERE request_id = $1 ORDER BY id", request_id)
        return [self._from_row(r) for r in rows]

    async def get(self, damage_id: str) -> DetectedDamage | None:
        row = await self._pool.fetchrow("SELECT * FROM detected_damages WHERE id = $1", damage_id)
        if row is None:
            return None
        return self._from_row(row)

    async def update(self, damage: DetectedDamage) -> None:
        logger.debug("UPDATE detected_damage id={}", damage.id)
        await self._pool.execute(
            "UPDATE detected_damages SET damage_type=$2, part_type=$3, is_deleted=$4 WHERE id=$1",
            damage.id,
            damage.damage_type.value,
            damage.part_type.value,
            damage.is_deleted,
        )

    async def soft_delete(self, damage_id: str) -> None:
        logger.debug("SOFT DELETE detected_damage id={}", damage_id)
        await self._pool.execute("UPDATE detected_damages SET is_deleted=TRUE WHERE id=$1", damage_id)

    @staticmethod
    def _from_row(row: asyncpg.Record) -> DetectedDamage:
        return DetectedDamage(
            id=str(row["id"]),
            request_id=str(row["request_id"]),
            part_id=str(row["part_id"]) if row["part_id"] else None,
            damage_type=DamageType(row["damage_type"]),
            part_type=PartType(row["part_type"]),
            source=DamageSource(row["source"]),
            confidence=row["confidence"],
            mask_image_key=row["mask_image_key"],
            is_deleted=row["is_deleted"],
        )
