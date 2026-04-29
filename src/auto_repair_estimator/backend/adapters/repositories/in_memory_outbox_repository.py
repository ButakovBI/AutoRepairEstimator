from __future__ import annotations

from datetime import UTC, datetime

from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.interfaces.outbox_repository import OutboxRepository


class InMemoryOutboxRepository(OutboxRepository):
    """In-memory outbox for dev/tests.

    Ordering of ``get_unpublished`` mirrors the Postgres repository:
    oldest-unpublished-first, deterministic by ``created_at`` then by
    insertion order for events with identical timestamps.
    """

    def __init__(self) -> None:
        self._events: dict[str, OutboxEvent] = {}
        self._order: list[str] = []

    async def add(self, event: OutboxEvent) -> None:
        self._events[event.id] = event
        self._order.append(event.id)

    async def get_unpublished(self, limit: int) -> list[OutboxEvent]:
        unpublished = [self._events[eid] for eid in self._order if self._events[eid].published_at is None]
        unpublished.sort(key=lambda e: e.created_at)
        return unpublished[:limit]

    async def mark_published(self, event_ids: list[str]) -> None:
        now = datetime.now(UTC)
        for eid in event_ids:
            event = self._events.get(eid)
            if event is None:
                continue
            self._events[eid] = OutboxEvent(
                id=event.id,
                aggregate_id=event.aggregate_id,
                topic=event.topic,
                payload=event.payload,
                created_at=event.created_at,
                published_at=now,
            )
