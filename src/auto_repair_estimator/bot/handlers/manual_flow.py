from __future__ import annotations

from typing import Any

from loguru import logger
from vkbottle import API
from vkbottle.bot import MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.keyboards.damage_edit import add_more_or_confirm_keyboard
from auto_repair_estimator.bot.keyboards.damage_type_selection import damage_type_selection_keyboard
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS
from auto_repair_estimator.bot.part_selection_send import send_part_selection_messages


async def handle_mode_selection(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    mode = payload["m"]
    peer_id = event.peer_id
    user_id = event.user_id

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
    part_type = payload["pt"]
    request_id = payload["rid"]
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
    request_id = payload["rid"]
    part_type = payload["pt"]
    damage_type = payload["dt"]

    try:
        await backend.add_damage(request_id=request_id, part_type=part_type, damage_type=damage_type)
    except Exception as exc:
        logger.error("Failed to add damage: {}", exc)
        await api.messages.send(peer_id=event.peer_id, message="Ошибка при добавлении повреждения.", random_id=0)
        return

    part_label = PART_LABELS.get(part_type, part_type)
    damage_label = DAMAGE_LABELS.get(damage_type, damage_type)
    await api.messages.send(
        peer_id=event.peer_id,
        message=f"Добавлено: {part_label} — {damage_label}",
        keyboard=add_more_or_confirm_keyboard(request_id),
        random_id=0,
    )


async def handle_add_more(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    request_id = payload["rid"]
    await send_part_selection_messages(
        api,
        event.peer_id,
        request_id,
        first_message="Выберите следующую повреждённую деталь:",
    )
