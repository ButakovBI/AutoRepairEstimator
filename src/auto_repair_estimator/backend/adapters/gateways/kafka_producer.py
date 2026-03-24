from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaProducer
from loguru import logger


class KafkaProducer:
    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            acks="all",
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await self._producer.start()
        logger.info("Kafka producer started (servers={})", self._bootstrap_servers)

    async def send(self, topic: str, message: dict[str, Any]) -> None:
        if self._producer is None:
            raise RuntimeError("KafkaProducer not started; call start() first")
        await self._producer.send_and_wait(topic, message)
        logger.debug("Published message to topic={}", topic)

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            logger.info("Kafka producer stopped")
