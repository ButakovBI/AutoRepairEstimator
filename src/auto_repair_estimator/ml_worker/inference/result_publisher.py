from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaProducer
from loguru import logger


class ResultPublisher:
    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            acks="all",
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await self._producer.start()
        logger.info("ResultPublisher started")

    async def publish_success(
        self,
        request_id: str,
        parts: list[dict[str, Any]],
        damages: list[dict[str, Any]],
        composited_image_key: str,
    ) -> None:
        payload: dict[str, Any] = {
            "request_id": request_id,
            "status": "success",
            "parts": parts,
            "damages": damages,
            "composited_image_key": composited_image_key,
            "error_message": None,
        }
        await self._send(payload)
        logger.info("Published success result for request_id={}", request_id)

    async def publish_error(self, request_id: str, error_message: str) -> None:
        payload: dict[str, Any] = {
            "request_id": request_id,
            "status": "error",
            "parts": [],
            "damages": [],
            "composited_image_key": None,
            "error_message": error_message,
        }
        await self._send(payload)
        logger.warning("Published error result for request_id={} error={}", request_id, error_message)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._producer is None:
            raise RuntimeError("ResultPublisher not started")
        await self._producer.send_and_wait(self._topic, payload)

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            logger.info("ResultPublisher stopped")
