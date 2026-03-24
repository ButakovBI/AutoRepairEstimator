from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from vkbottle import GroupEventType
from vkbottle.bot import Bot, Message, MessageEvent
from vkbottle.exception_factory.base_exceptions import VKAPIError

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.config import get_config
from auto_repair_estimator.bot.handlers.damage_edit import handle_edit_action, handle_edit_damage_type
from auto_repair_estimator.bot.handlers.manual_flow import (
    handle_add_more,
    handle_damage_type_selection,
    handle_mode_selection,
    handle_part_selection,
)
from auto_repair_estimator.bot.handlers.photo import handle_photo
from auto_repair_estimator.bot.handlers.pricing import handle_confirm
from auto_repair_estimator.bot.handlers.start import handle_start
from auto_repair_estimator.bot.notification_consumer import NotificationConsumer


def _create_bot() -> tuple[Bot, BackendClient]:
    config = get_config()
    bot = Bot(token=config.vk_group_token)
    backend = BackendClient(config.backend_url)
    return bot, backend


def _register_handlers(bot: Bot, backend: BackendClient) -> None:
    @bot.on.message(text=["/start", "Начать", "начать"])
    async def on_start(message: Message) -> None:
        await handle_start(message)

    @bot.on.message()
    async def on_message(message: Message) -> None:
        photos = message.get_photo_attachments()
        if photos:
            await handle_photo(message, backend, bot.api)
            return
        await message.answer("Напишите /start или 'Начать', чтобы начать оценку стоимости ремонта.")

    @bot.on.raw_event(GroupEventType.MESSAGE_EVENT, dataclass=MessageEvent)
    async def on_callback(event: MessageEvent) -> None:
        payload: dict[str, Any] = event.get_payload_json() or {}
        cmd = payload.get("cmd")

        await event.show_snackbar("...")

        if cmd == "mode":
            await handle_mode_selection(event, payload, backend, bot.api)
        elif cmd == "part":
            await handle_part_selection(event, payload, backend, bot.api)
        elif cmd == "dmg":
            await handle_damage_type_selection(event, payload, backend, bot.api)
        elif cmd == "edit":
            await handle_edit_action(event, payload, backend, bot.api)
        elif cmd == "edtype":
            await handle_edit_damage_type(event, payload, backend, bot.api)
        elif cmd == "confirm":
            await handle_confirm(event, payload, backend, bot.api)
        elif cmd == "addmore":
            await handle_add_more(event, payload, backend, bot.api)
        else:
            logger.warning("Unknown callback cmd={}", cmd)


async def main() -> None:
    config = get_config()
    bot, backend = _create_bot()
    _register_handlers(bot, backend)

    notification_consumer = NotificationConsumer(
        bootstrap_servers=config.kafka_bootstrap_servers,
        topic=config.kafka_topic_notifications,
        api=bot.api,
        backend=backend,
        s3_endpoint=config.s3_endpoint,
        s3_access_key=config.s3_access_key,
        s3_secret_key=config.s3_secret_key,
    )

    async def run_consumer() -> None:
        try:
            await notification_consumer.run()
        except Exception as exc:
            logger.error("NotificationConsumer crashed: {}", exc)

    logger.info("Starting VK bot")
    consumer_task = asyncio.create_task(run_consumer(), name="notification-consumer")

    # vkbottle BaseFramework.run_polling: if loop_wrapper.is_running is False it calls
    # LoopWrapper.run(), which raises when asyncio already has a running loop (e.g. asyncio.run).
    # Mark wrapper as running so run_polling uses await polling() on the current loop.
    bot.loop_wrapper._running = True  # noqa: SLF001

    try:
        await bot.run_polling()
    except VKAPIError as exc:
        err = str(exc).lower()
        if exc.code == 100 and "longpoll" in err:
            logger.error(
                "VK API: для сообщества не включён Long Poll API (groups.getLongPollServer). "
                "ВКонтакте: управление сообществом → Настройки → Работа с API → "
                "включите «Long Poll API», сохраните и перезапустите контейнер бота."
            )
        raise
    finally:
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
