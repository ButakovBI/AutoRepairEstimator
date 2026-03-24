from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from aiokafka import AIOKafkaConsumer
from loguru import logger


class KafkaConsumer:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode()),
        )
        await self._consumer.start()
        logger.info("Kafka consumer started topic={} group={}", self._topic, self._group_id)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        if self._consumer is None:
            raise RuntimeError("KafkaConsumer not started; call start() first")
        async for msg in self._consumer:
            yield msg.value

    async def stop(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()
            logger.info("Kafka consumer stopped topic={}", self._topic)
