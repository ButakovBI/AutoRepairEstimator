from __future__ import annotations

import asyncio

from loguru import logger

from auto_repair_estimator.backend.adapters.gateways.kafka_producer import KafkaProducer
from auto_repair_estimator.backend.domain.interfaces.outbox_repository import OutboxRepository


class OutboxFlusher:
    def __init__(
        self,
        outbox_repository: OutboxRepository,
        kafka_producer: KafkaProducer,
        poll_interval_ms: int = 500,
        batch_size: int = 10,
    ) -> None:
        self._outbox = outbox_repository
        self._producer = kafka_producer
        self._poll_interval = poll_interval_ms / 1000.0
        self._batch_size = batch_size

    async def run(self) -> None:
        logger.info("OutboxFlusher started (interval={}s batch={})", self._poll_interval, self._batch_size)
        while True:
            try:
                await self._flush_batch()
            except asyncio.CancelledError:
                logger.info("OutboxFlusher cancelled")
                return
            except Exception as exc:
                logger.error("OutboxFlusher error: {}", exc)
            await asyncio.sleep(self._poll_interval)

    async def _flush_batch(self) -> None:
        events = await self._outbox.get_unpublished(self._batch_size)
        if not events:
            return

        published_ids: list[str] = []
        for event in events:
            try:
                await self._producer.send(event.topic, event.payload)
                published_ids.append(event.id)
            except Exception as exc:
                logger.error("Failed to publish outbox event id={}: {}", event.id, exc)

        if published_ids:
            await self._outbox.mark_published(published_ids)
            logger.debug("Flushed {} outbox events", len(published_ids))
