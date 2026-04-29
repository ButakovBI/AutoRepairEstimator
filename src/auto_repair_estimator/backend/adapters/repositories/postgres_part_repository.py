from __future__ import annotations

import asyncpg
from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_part import DetectedPart
from auto_repair_estimator.backend.domain.value_objects.request_enums import PartType


class PostgresPartRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add(self, part: DetectedPart) -> None:
        logger.debug("INSERT detected_part id={} request_id={}", part.id, part.request_id)
        await self._pool.execute(
            """
            INSERT INTO detected_parts
                (id, request_id, part_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, crop_image_key)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            part.id,
            part.request_id,
            part.part_type.value,
            part.confidence,
            part.x,
            part.y,
            part.width,
            part.height,
            part.crop_image_key,
        )

    async def get_by_request_id(self, request_id: str) -> list[DetectedPart]:
        rows = await self._pool.fetch("SELECT * FROM detected_parts WHERE request_id = $1 ORDER BY id", request_id)
        return [self._from_row(r) for r in rows]

    @staticmethod
    def _from_row(row: asyncpg.Record) -> DetectedPart:
        return DetectedPart(
            id=str(row["id"]),
            request_id=str(row["request_id"]),
            part_type=PartType(row["part_type"]),
            confidence=row["confidence"],
            x=row["bbox_x"] or 0.0,
            y=row["bbox_y"] or 0.0,
            width=row["bbox_w"] or 0.0,
            height=row["bbox_h"] or 0.0,
            crop_image_key=row["crop_image_key"],
        )
