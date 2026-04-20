from __future__ import annotations

from typing import Any

from loguru import logger
from vkbottle import API, PhotoMessageUploader
from vkbottle.bot import MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.damage_list_format import format_damage_list
from auto_repair_estimator.bot.keyboards.damage_edit import (
    damage_edit_keyboard,
    edit_damage_type_keyboard,
    inference_result_keyboard,
)
from auto_repair_estimator.bot.part_selection_send import send_part_selection_messages


def _format_damages_text(damages: list[dict[str, Any]]) -> str:
    # Retained for the "inference complete" card so that text header stays
    # user-facing ("Обнаруженные повреждения"); edit/manual flows use the
    # more-generic ``format_damage_list`` helper directly.
    return format_damage_list(damages, header="Обнаруженные повреждения:")


async def send_inference_result(
    api: API,
    peer_id: int,
    request_id: str,
    damages: list[dict[str, Any]],
    composited_image_key: str | None,
    backend: BackendClient,
    s3_endpoint: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
) -> None:
    if composited_image_key and s3_endpoint and s3_access_key and s3_secret_key:
        try:
            import httpx

            from auto_repair_estimator.backend.adapters.gateways.minio_storage_gateway import MinioStorageGateway

            storage = MinioStorageGateway(s3_endpoint, s3_access_key, s3_secret_key)
            url = await storage.generate_presigned_get_url(composited_image_key)
            async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                resp.raise_for_status()
                image_bytes = resp.content

            uploader = PhotoMessageUploader(api)
            attachment = await uploader.upload(file_source=image_bytes, peer_id=peer_id)
            await api.messages.send(peer_id=peer_id, attachment=attachment, random_id=0)
        except Exception as exc:
            logger.warning("Could not send composited image: {}", exc)

    if damages:
        text = _format_damages_text(damages)
        await api.messages.send(
            peer_id=peer_id,
            message=text + "\n\nМодель выдала результат корректно?",
            keyboard=inference_result_keyboard(request_id),
            random_id=0,
        )
    else:
        await send_part_selection_messages(
            api,
            peer_id,
            request_id,
            first_message="Не удалось обнаружить повреждения. Укажите их вручную:",
        )


async def handle_edit_action(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    action = payload.get("a")
    request_id = payload.get("rid")
    if not action or not request_id:
        logger.warning("handle_edit_action received malformed payload: {}", payload)
        await api.messages.send(peer_id=event.peer_id, message="Некорректная кнопка. Напишите /start, чтобы начать заново.", random_id=0)
        return
    damage_id = payload.get("did", "")

    if action == "start_edit":
        try:
            data = await backend.get_request(request_id)
            damages = data.get("damages", [])
        except Exception as exc:
            logger.error("Failed to get request: {}", exc)
            await api.messages.send(peer_id=event.peer_id, message="Ошибка получения данных.", random_id=0)
            return
        await api.messages.send(
            peer_id=event.peer_id,
            message="Редактирование повреждений:",
            keyboard=damage_edit_keyboard(request_id, damages),
            random_id=0,
        )

    elif action == "edit_type":
        # ``pt`` is propagated through the payload by damage_edit_keyboard so
        # we don't need a backend round-trip just to look up the current part.
        # Legacy payloads without ``pt`` default to an empty string, which
        # edit_damage_type_keyboard degrades to "show every type".
        part_type_value = payload.get("pt", "")
        # Embed the full damage list in the header (bug #5): users complained
        # they had to scroll back to the previous message to remember what
        # else the model had found while they were editing a single item.
        list_text = await _format_current_damage_list(backend, request_id)
        await api.messages.send(
            peer_id=event.peer_id,
            message=f"{list_text}\n\nВыберите новый тип повреждения:",
            keyboard=edit_damage_type_keyboard(request_id, damage_id, part_type_value),
            random_id=0,
        )

    elif action == "delete":
        try:
            await backend.delete_damage(request_id, damage_id)
            data = await backend.get_request(request_id)
            active_damages = [d for d in data.get("damages", []) if not d.get("is_deleted", False)]
            await api.messages.send(
                peer_id=event.peer_id,
                message="Повреждение удалено.",
                keyboard=damage_edit_keyboard(request_id, active_damages),
                random_id=0,
            )
        except Exception as exc:
            logger.error("Failed to delete damage: {}", exc)
            await api.messages.send(peer_id=event.peer_id, message="Ошибка при удалении повреждения.", random_id=0)


async def handle_edit_damage_type(
    event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API
) -> None:
    request_id = payload.get("rid")
    damage_id = payload.get("did")
    damage_type = payload.get("dt")
    if not request_id or not damage_id or not damage_type:
        logger.warning("handle_edit_damage_type received malformed payload: {}", payload)
        await api.messages.send(peer_id=event.peer_id, message="Некорректная кнопка. Напишите /start, чтобы начать заново.", random_id=0)
        return

    try:
        await backend.edit_damage(request_id, damage_id, damage_type)
        data = await backend.get_request(request_id)
        active_damages = [d for d in data.get("damages", []) if not d.get("is_deleted", False)]
        header = format_damage_list(active_damages, header="Тип повреждения обновлён.\n\nТекущий список:")
        await api.messages.send(
            peer_id=event.peer_id,
            message=header,
            keyboard=damage_edit_keyboard(request_id, active_damages),
            random_id=0,
        )
    except Exception as exc:
        logger.error("Failed to edit damage: {}", exc)
        await api.messages.send(peer_id=event.peer_id, message="Ошибка при изменении повреждения.", random_id=0)


async def handle_back_edit(
    event: MessageEvent,
    payload: dict[str, Any],
    backend: BackendClient,
    api: API,
) -> None:
    """Return the user to the top-level damage-edit list.

    Used by the "← К списку повреждений" button on the edit-damage-type
    sub-menu. We refetch the damage list so the user sees the current
    state of the basket, not whatever snapshot was encoded in the
    previous keyboard payload.
    """

    request_id = payload.get("rid")
    if not request_id:
        logger.warning("handle_back_edit received malformed payload: {}", payload)
        await api.messages.send(
            peer_id=event.peer_id,
            message="Некорректная кнопка. Нажмите «Начать», чтобы начать заново.",
            random_id=0,
        )
        return
    rid = str(request_id)
    try:
        data = await backend.get_request(rid)
    except Exception as exc:
        logger.error("Failed to refetch damages for back_edit rid={}: {}", rid, exc)
        await api.messages.send(
            peer_id=event.peer_id,
            message="Ошибка получения данных. Попробуйте ещё раз.",
            random_id=0,
        )
        return
    active_damages = [d for d in data.get("damages", []) if not d.get("is_deleted", False)]
    body = format_damage_list(active_damages, header="Редактирование повреждений:")
    await api.messages.send(
        peer_id=event.peer_id,
        message=body,
        keyboard=damage_edit_keyboard(rid, active_damages),
        random_id=0,
    )


async def _format_current_damage_list(
    backend: BackendClient, request_id: str
) -> str:
    """Fetch active damages and render them as a numbered list string.

    Isolated helper so the edit-type entry point can reuse the same
    formatting the manual-add flow uses. On backend failure we degrade to
    a terse placeholder — the failure is already logged by the caller.
    """

    try:
        data = await backend.get_request(request_id)
    except Exception as exc:
        logger.warning(
            "Could not refetch damages for edit header rid={}: {}", request_id, exc
        )
        return "Текущий список повреждений временно недоступен."
    active = [d for d in data.get("damages", []) if not d.get("is_deleted", False)]
    return format_damage_list(active, header="Текущие повреждения:")
