from __future__ import annotations

from typing import Any

from loguru import logger
from vkbottle import API
from vkbottle.bot import MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.damage_list_format import format_damage_list
from auto_repair_estimator.bot.keyboards.damage_edit import add_more_or_confirm_keyboard
from auto_repair_estimator.bot.keyboards.damage_type_selection import damage_type_selection_keyboard
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS
from auto_repair_estimator.bot.part_selection_send import send_part_selection_messages
from auto_repair_estimator.bot.session_lifecycle import abandon_active_session


async def handle_mode_selection(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    mode = payload.get("m")
    peer_id = event.peer_id
    user_id = event.user_id
    if mode not in ("ml", "manual"):
        logger.warning("handle_mode_selection received invalid payload: {}", payload)
        await api.messages.send(peer_id=peer_id, message="Некорректная кнопка. Напишите /start, чтобы начать заново.", random_id=0)
        return

    # Picking a mode IS a session reset — any non-terminal request the
    # chat might still have (e.g. an ML request in PROCESSING, a manual
    # session reached via an old mode-selection keyboard) must be closed
    # before the new one is created. The notice the user already saw on
    # the welcome screen of ``handle_start`` / ``handle_start_callback``
    # covers this case; we don't emit a second one to avoid double-noise.
    await abandon_active_session(backend, peer_id)

    try:
        data = await backend.create_request(chat_id=peer_id, user_id=user_id, mode=mode)
    except Exception as exc:
        logger.error("Failed to create request: {}", exc)
        await api.messages.send(peer_id=peer_id, message="Произошла ошибка. Попробуйте ещё раз.", random_id=0)
        return

    request_id = data["id"]

    if mode == "manual":
        await send_part_selection_messages(
            api,
            peer_id,
            request_id,
            first_message="Выберите повреждённую деталь:",
        )
    else:
        await api.messages.send(
            peer_id=peer_id,
            message="Отправьте чёткую фотографию повреждения автомобиля.",
            random_id=0,
        )


async def handle_part_selection(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    part_type = payload.get("pt")
    request_id = payload.get("rid")
    if not part_type or not request_id:
        logger.warning("handle_part_selection received malformed payload: {}", payload)
        await api.messages.send(peer_id=event.peer_id, message="Некорректная кнопка. Напишите /start, чтобы начать заново.", random_id=0)
        return
    part_label = PART_LABELS.get(part_type, part_type)

    await api.messages.send(
        peer_id=event.peer_id,
        message=f"Деталь: {part_label}\nВыберите тип повреждения:",
        keyboard=damage_type_selection_keyboard(request_id, part_type),
        random_id=0,
    )


async def handle_damage_type_selection(
    event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API
) -> None:
    request_id = payload.get("rid")
    part_type = payload.get("pt")
    damage_type = payload.get("dt")
    if not request_id or not part_type or not damage_type:
        logger.warning("handle_damage_type_selection received malformed payload: {}", payload)
        await api.messages.send(peer_id=event.peer_id, message="Некорректная кнопка. Напишите /start, чтобы начать заново.", random_id=0)
        return

    try:
        result = await backend.add_damage(
            request_id=request_id, part_type=part_type, damage_type=damage_type
        )
    except Exception as exc:
        logger.error("Failed to add damage: {}", exc)
        await api.messages.send(peer_id=event.peer_id, message="Ошибка при добавлении повреждения.", random_id=0)
        return

    part_label = PART_LABELS.get(part_type, part_type)
    damage_label = DAMAGE_LABELS.get(damage_type, damage_type)
    # Re-read the full basket from the backend and embed it in the reply
    # (bug #5). This is one extra round-trip per add, but it's the only
    # way to keep the list accurate: the bot does not cache damages and
    # any previous delete/edit could have changed the set. The cost is
    # negligible compared to the VK send latency that dominates here.
    running_list = await _running_damage_list_or_empty(backend, request_id)
    # ``already_existed=True`` means the backend short-circuited the add
    # because the same pair was already active on the request (domain
    # invariant: one active damage per ``(part, type)``). We switch the
    # heading from "Добавлено" to a softer "уже есть" so the user knows
    # the tap was recognised but didn't grow the list — which matches
    # the state of the basket.
    already_existed = bool(result.get("already_existed")) if isinstance(result, dict) else False
    heading = (
        f"Это повреждение уже есть в списке: {part_label} — {damage_label}"
        if already_existed
        else f"Добавлено: {part_label} — {damage_label}"
    )
    await api.messages.send(
        peer_id=event.peer_id,
        message=f"{heading}\n\n{running_list}",
        keyboard=add_more_or_confirm_keyboard(request_id),
        random_id=0,
    )


async def _running_damage_list_or_empty(backend: BackendClient, request_id: str) -> str:
    """Return the formatted running list or a safe fallback on error.

    A backend hiccup here must not block the "Добавить ещё / Подтвердить"
    keyboard from reaching the user — the add already succeeded server-
    side. We degrade to a terse placeholder in that case.
    """

    try:
        data = await backend.get_request(request_id)
    except Exception as exc:
        logger.warning("Could not re-fetch damages for request {}: {}", request_id, exc)
        return "Текущий список повреждений временно недоступен."
    damages = data.get("damages", []) if isinstance(data, dict) else []
    return format_damage_list(damages)


async def handle_add_more(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    request_id = payload.get("rid")
    if not request_id:
        logger.warning("handle_add_more received malformed payload: {}", payload)
        await api.messages.send(peer_id=event.peer_id, message="Некорректная кнопка. Напишите /start, чтобы начать заново.", random_id=0)
        return
    await send_part_selection_messages(
        api,
        event.peer_id,
        request_id,
        first_message="Выберите следующую повреждённую деталь:",
    )


async def handle_back_parts(
    event: MessageEvent,
    payload: dict[str, Any],
    backend: BackendClient,  # noqa: ARG001 - uniform handler signature
    api: API,
) -> None:
    """Re-present the part-selection keyboard.

    Triggered from the "← К выбору детали" button on the damage-type
    screen (bug #3). We don't undo the damage the user was about to add —
    they haven't clicked a damage type yet, so there is nothing to undo.
    """

    request_id = payload.get("rid")
    if not request_id:
        logger.warning("handle_back_parts received malformed payload: {}", payload)
        await api.messages.send(
            peer_id=event.peer_id,
            message="Некорректная кнопка. Нажмите «Начать», чтобы начать заново.",
            random_id=0,
        )
        return
    await send_part_selection_messages(
        api,
        event.peer_id,
        str(request_id),
        first_message="Выберите повреждённую деталь:",
    )
