from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

# Moscow time — fixed offset because the operator team and every user
# of this bot lives in GMT+3. Relying on the host's local timezone
# (``datetime.astimezone()`` with no argument) would render UTC inside
# the production container, so a request created at 14:00 MSK showed
# up in the notification as "11:00" — the user's complaint that led to
# this change.
MOSCOW_TZ = timezone(timedelta(hours=3), name="MSK")

from loguru import logger
from vkbottle import API

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.damage_edit import send_inference_result
from auto_repair_estimator.bot.keyboards.start import start_keyboard
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
            # Pass the worker-supplied reason through to the user when we
            # have one. ``no_parts_detected`` is by far the most common
            # ("I couldn't find any car parts in this photo") and the
            # actionable advice is different from a generic crash — so
            # we branch on the reason instead of rendering one-size-fits-
            # all copy.
            reason = (message.get("error_message") or "").strip()
            if reason == "no_parts_detected":
                first_message = (
                    "Я не нашёл на фото ни одной детали автомобиля. Пришлите, "
                    "пожалуйста, более чёткий снимок, на котором повреждение "
                    "видно целиком, или укажите повреждения вручную:"
                )
            else:
                first_message = (
                    "Не удалось обработать изображение автоматически. "
                    "Укажите повреждения вручную:"
                )
            await send_part_selection_messages(
                self._api,
                peer_id,
                str(request_id),
                first_message=first_message,
            )
            # Always attach the Start keyboard after the failure path:
            # the user's other escape hatch (sending another photo) is
            # easy to miss in the middle of a manual-entry screen, so we
            # offer a zero-typing way to restart the whole flow.
            await self._api.messages.send(
                peer_id=peer_id,
                message=(
                    "Если хотите прислать другое фото — нажмите «Начать» "
                    "и выберите режим «С фотографией (ML)»."
                ),
                keyboard=start_keyboard(),
                random_id=0,
            )

        elif notification_type == "request_timeout":
            created_at_iso = message.get("request_created_at")
            text = self._format_timeout_message(created_at_iso)
            await self._api.messages.send(
                peer_id=peer_id,
                message=text,
                keyboard=start_keyboard(),
                random_id=0,
            )

        else:
            logger.warning("Unknown notification type={}", notification_type)

    @staticmethod
    def _format_timeout_message(created_at_iso: Any) -> str:
        """Render the user-facing timeout text, including when the request was created.

        The backend sends ``request_created_at`` as ISO-8601 UTC. We always
        convert to Moscow time (UTC+3) — the previous "local machine"
        conversion rendered UTC inside the container and confused users
        who expected MSK. If the timestamp is missing or malformed we
        degrade to the timeless variant instead of raising, because a
        broken side-channel must not swallow the whole notification.

        The copy is deliberately framed as a "timeout — session closed"
        rather than "finished with an error". The user did nothing
        wrong: either the model took too long or our infrastructure
        couldn't deliver a result in time, and calling that "an error"
        shifts blame onto them.
        """

        base = (
            "Время ожидания обработки вашего запроса истекло — заявка закрыта.\n"
            "Чтобы попробовать ещё раз, нажмите «Начать»."
        )
        if not isinstance(created_at_iso, str) or not created_at_iso:
            return base
        try:
            dt = datetime.fromisoformat(created_at_iso)
        except ValueError:
            return base
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        moscow_dt = dt.astimezone(MOSCOW_TZ)
        when = moscow_dt.strftime("%d.%m %H:%M")
        return (
            f"Время ожидания обработки вашего запроса от {when} (МСК) истекло — "
            "заявка закрыта.\n"
            "Чтобы попробовать ещё раз, нажмите «Начать»."
        )
