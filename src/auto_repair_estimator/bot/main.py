from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from vkbottle import GroupEventType
from vkbottle.bot import Bot, Message, MessageEvent
from vkbottle.exception_factory.base_exceptions import VKAPIError

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.config import get_config
from auto_repair_estimator.bot.handlers.damage_edit import (
    handle_back_edit,
    handle_edit_action,
    handle_edit_damage_type,
    handle_group_action,
)
from auto_repair_estimator.bot.handlers.manual_flow import (
    handle_add_more,
    handle_back_parts,
    handle_damage_type_selection,
    handle_mode_selection,
    handle_part_selection,
)
from auto_repair_estimator.bot.handlers.photo import handle_photo
from auto_repair_estimator.bot.handlers.pricing import handle_confirm
from auto_repair_estimator.bot.handlers.start import (
    active_session_nudge,
    handle_start,
    handle_start_callback,
    send_no_active_session_reply,
)
from auto_repair_estimator.bot.keyboards.start import start_keyboard
from auto_repair_estimator.bot.notification_consumer import NotificationConsumer


def _create_bot() -> tuple[Bot, BackendClient]:
    config = get_config()
    bot = Bot(token=config.vk_group_token)
    backend = BackendClient(config.backend_url)
    return bot, backend


# Callback-dispatch policy tables. Module-level so unit tests can assert
# on the invariant set without spinning up a full Bot.
CALLBACK_HANDLERS: dict[str, Any] = {
    "start": handle_start_callback,
    "mode": handle_mode_selection,
    "part": handle_part_selection,
    "dmg": handle_damage_type_selection,
    "edit": handle_edit_action,
    "edtype": handle_edit_damage_type,
    "grp": handle_group_action,
    "confirm": handle_confirm,
    "addmore": handle_add_more,
    "back_parts": handle_back_parts,
    "back_edit": handle_back_edit,
}
# Cmds that carry a pre-existing ``rid`` in the payload. Before dispatch
# we verify: (1) the chat has any active session, (2) its id matches
# the payload ``rid`` (guard against stale buttons from a prior
# session — bug #3), (3) if the cmd is a manual-flow mutation, the
# status is PRICING (ML sessions in CREATED/QUEUED/PROCESSING must
# finish their analysis first).
CMDS_REQUIRING_ACTIVE_RID: frozenset[str] = frozenset(
    {"part", "dmg", "edit", "edtype", "grp", "confirm", "addmore", "back_parts", "back_edit"}
)
CMDS_REQUIRING_PRICING_STATUS: frozenset[str] = frozenset({"part", "dmg", "addmore", "back_parts"})


async def handle_incoming_message(
    backend: BackendClient,
    api: Any,
    message: Message,
) -> None:
    """Top-level dispatcher for free-form user messages (non-callback path).

    Pulled out of the ``on_message`` closure so unit tests can drive it
    directly with mock ``BackendClient`` / ``vkbottle.API`` fakes — the
    full ``_register_handlers`` wiring requires a real ``Bot`` token and
    is awkward to exercise end-to-end in a unit test. Behaviour:

    * **Photo branch.** Any photo is forwarded to ``handle_photo``
      which enforces the single-active-session invariant itself
      (abandons the prior session, if any, with a visible notice) —
      the dispatcher intentionally stays out of the decision so the
      abandonment logic has exactly one home.
    * **Text branch.** No active session → invite the user to press
      «Начать». Active session → render the ``(mode, status)``-aware
      nudge so we never tell the user "use the buttons" when the last
      message had none.
    """

    photos = message.get_photo_attachments()
    if photos:
        await handle_photo(message, backend, api)
        return

    try:
        active = await backend.get_active_request(message.peer_id)
    except Exception as exc:
        logger.warning("Could not probe active request for peer_id={}: {}", message.peer_id, exc)
        active = None

    if active is None:
        await send_no_active_session_reply(api, message.peer_id)
        return

    nudge = active_session_nudge(active.get("mode"), active.get("status"))
    await message.answer(nudge, keyboard=start_keyboard())


async def validate_active_rid_for_callback(
    backend: BackendClient,
    api: Any,
    peer_id: int,
    payload: dict[str, Any],
    cmd: str,
) -> bool:
    """Return ``True`` iff the callback may proceed, else send a nudge.

    Enforces the three invariants documented on ``CMDS_REQUIRING_ACTIVE_RID``.
    The function is module-level so unit tests can exercise each branch
    in isolation (malformed payload, missing active, rid mismatch, wrong
    status) without reconstructing the ``on_callback`` closure. Side-
    effect: emits exactly one VK message on the rejection branch.
    """

    rid = payload.get("rid")
    if not rid:
        await send_no_active_session_reply(api, peer_id)
        return False

    try:
        active = await backend.get_active_request(peer_id)
    except Exception as exc:
        logger.warning(
            "Active-request probe failed for peer_id={} rid={} err={}",
            peer_id,
            rid,
            exc,
        )
        await send_no_active_session_reply(api, peer_id)
        return False
    if active is None:
        await send_no_active_session_reply(api, peer_id)
        return False

    if str(active.get("id")) != str(rid):
        logger.info(
            "Rejecting stale callback cmd={} payload_rid={} active_rid={}",
            cmd,
            rid,
            active.get("id"),
        )
        await send_no_active_session_reply(api, peer_id)
        return False

    status = (active.get("status") or "").lower()
    if cmd in CMDS_REQUIRING_PRICING_STATUS and status != "pricing":
        text = active_session_nudge(active.get("mode"), active.get("status"))
        await api.messages.send(peer_id=peer_id, message=text, keyboard=start_keyboard(), random_id=0)
        return False
    return True


def _register_handlers(bot: Bot, backend: BackendClient) -> None:
    @bot.on.message(text=["/start", "Начать", "начать"])
    async def on_start(message: Message) -> None:
        await handle_start(message, backend)

    @bot.on.message()
    async def on_message(message: Message) -> None:
        await handle_incoming_message(backend, bot.api, message)

    @bot.on.raw_event(GroupEventType.MESSAGE_EVENT, dataclass=MessageEvent)
    async def on_callback(event: MessageEvent) -> None:
        payload: dict[str, Any] = event.get_payload_json() or {}
        cmd = payload.get("cmd")

        # Acknowledge the callback silently so VK removes the button's
        # loading spinner. Previously we called ``show_snackbar("...")``,
        # which literally displayed the three dots as a toast next to
        # every press — users reported it as visual noise. An empty
        # answer dismisses the spinner without any UI side-effect. Wrapped
        # in try/except because a transient VK hiccup here must not
        # block the main handler: at worst VK will time out the spinner
        # on its own after ~10s.
        try:
            await event.send_empty_answer()
        except Exception as exc:
            logger.warning("send_empty_answer failed for cmd={}: {}", cmd, exc)

        handler = CALLBACK_HANDLERS.get(cmd)
        if handler is None:
            logger.warning("Unknown callback cmd={}", cmd)
            await send_no_active_session_reply(bot.api, event.peer_id)
            return

        if cmd in CMDS_REQUIRING_ACTIVE_RID and not await validate_active_rid_for_callback(
            backend, bot.api, event.peer_id, payload, cmd
        ):
            return

        # Single outer try/except: any unhandled exception inside a handler
        # (malformed payload, backend timeout, VK API hiccup) must not escape
        # into vkbottle -- otherwise the snackbar stays active and the user
        # never receives a reply. We send a generic fallback message instead.
        try:
            await handler(event, payload, backend, bot.api)
        except Exception as exc:
            logger.exception("Callback handler {} failed on payload={}: {}", cmd, payload, exc)
            try:
                await bot.api.messages.send(
                    peer_id=event.peer_id,
                    message="Произошла ошибка. Попробуйте ещё раз.",
                    random_id=0,
                )
            except Exception as reply_exc:  # pragma: no cover - best-effort
                logger.error("Could not send fallback error reply: {}", reply_exc)


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
