from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from vkbottle import API

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.damage_edit import send_inference_result
from auto_repair_estimator.bot.part_selection_send import send_part_selection_messages


class NotificationConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        api: API,
        backend: BackendClient,
        s3_endpoint: str | None = None,
        s3_access_key: str | None = None,
        s3_secret_key: str | None = None,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._api = api
        self._backend = backend
        self._s3_endpoint = s3_endpoint
        self._s3_access_key = s3_access_key
        self._s3_secret_key = s3_secret_key

    async def run(self) -> None:
        import json

        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id="bot-notifications",
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode()),
        )
        await consumer.start()
        logger.info("NotificationConsumer started topic={}", self._topic)
        try:
            async for msg in consumer:
                try:
                    await self._handle(msg.value)
                except Exception as exc:
                    logger.error("NotificationConsumer error: {}", exc)
        except asyncio.CancelledError:
            logger.info("NotificationConsumer cancelled")
            raise
        finally:
            await consumer.stop()

    async def _handle(self, message: dict[str, Any]) -> None:
        chat_id = message.get("chat_id")
        request_id = message.get("request_id")
        notification_type = message.get("type")

        if not chat_id or not request_id:
            logger.warning("Invalid notification message: missing chat_id or request_id")
            return

        peer_id = int(chat_id)

        if notification_type == "inference_complete":
            damages = message.get("damages", [])
            composited_image_key = message.get("composited_image_key")
            await send_inference_result(
                api=self._api,
                peer_id=peer_id,
                request_id=str(request_id),
                damages=damages,
                composited_image_key=composited_image_key,
                backend=self._backend,
                s3_endpoint=self._s3_endpoint,
                s3_access_key=self._s3_access_key,
                s3_secret_key=self._s3_secret_key,
            )

        elif notification_type == "inference_failed":
            await send_part_selection_messages(
                self._api,
                peer_id,
                str(request_id),
                first_message=("Не удалось обработать изображение автоматически. Укажите повреждения вручную:"),
            )

        elif notification_type == "request_timeout":
            await self._api.messages.send(
                peer_id=peer_id,
                message=(
                    "Обработка вашего запроса завершилась с ошибкой (превышено время ожидания).\n"
                    "Попробуйте ещё раз или используйте ручной ввод (напишите /start)."
                ),
                random_id=0,
            )

        else:
            logger.warning("Unknown notification type={}", notification_type)
