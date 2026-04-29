from typing import Protocol

from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent


class OutboxRepository(Protocol):
    async def add(self, event: OutboxEvent) -> None: ...

    async def get_unpublished(self, limit: int) -> list[OutboxEvent]: ...

    async def mark_published(self, event_ids: list[str]) -> None: ...
