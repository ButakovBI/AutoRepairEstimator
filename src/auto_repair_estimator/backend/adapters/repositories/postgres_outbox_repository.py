from __future__ import annotations

import json
from datetime import UTC, datetime

import asyncpg
from loguru import logger

from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent


class PostgresOutboxRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add(self, event: OutboxEvent) -> None:
        logger.debug("INSERT outbox_event id={} topic={}", event.id, event.topic)
        await self._pool.execute(
            """
            INSERT INTO outbox_events (id, aggregate_id, topic, payload, created_at)
            VALUES ($1,$2,$3,$4,$5)
            """,
            event.id,
            event.aggregate_id,
            event.topic,
            json.dumps(event.payload),
            event.created_at,
        )

    async def get_unpublished(self, limit: int) -> list[OutboxEvent]:
        rows = await self._pool.fetch(
            """
            SELECT * FROM outbox_events
            WHERE published_at IS NULL
            ORDER BY created_at
            LIMIT $1
            """,
            limit,
        )
        return [self._from_row(r) for r in rows]

    async def mark_published(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        logger.debug("Marking {} outbox events as published", len(event_ids))
        await self._pool.execute(
            "UPDATE outbox_events SET published_at=$1 WHERE id = ANY($2::uuid[])",
            datetime.now(UTC),
            event_ids,
        )

    @staticmethod
    def _from_row(row: asyncpg.Record) -> OutboxEvent:
        return OutboxEvent(
            id=str(row["id"]),
            aggregate_id=str(row["aggregate_id"]),
            topic=row["topic"],
            payload=json.loads(row["payload"]),
            created_at=row["created_at"],
            published_at=row["published_at"],
        )
