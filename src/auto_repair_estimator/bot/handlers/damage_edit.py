from __future__ import annotations

from typing import Any

from loguru import logger
from vkbottle import API, PhotoMessageUploader
from vkbottle.bot import MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.keyboards.damage_edit import (
    damage_edit_keyboard,
    edit_damage_type_keyboard,
    inference_result_keyboard,
)
from auto_repair_estimator.bot.part_selection_send import send_part_selection_messages
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


def _format_damages_text(damages: list[dict[str, Any]]) -> str:
    lines = ["Обнаруженные повреждения:"]
    for i, d in enumerate(damages, 1):
        part_label = PART_LABELS.get(d.get("part_type", ""), d.get("part_type", "?"))
        damage_label = DAMAGE_LABELS.get(d.get("damage_type", ""), d.get("damage_type", "?"))
        lines.append(f"{i}. {part_label} — {damage_label}")
    return "\n".join(lines)


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
    action = payload["a"]
    request_id = payload["rid"]
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
        await api.messages.send(
            peer_id=event.peer_id,
            message="Выберите новый тип повреждения:",
            keyboard=edit_damage_type_keyboard(request_id, damage_id),
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
    request_id = payload["rid"]
    damage_id = payload["did"]
    damage_type = payload["dt"]

    try:
        await backend.edit_damage(request_id, damage_id, damage_type)
        data = await backend.get_request(request_id)
        active_damages = [d for d in data.get("damages", []) if not d.get("is_deleted", False)]
        await api.messages.send(
            peer_id=event.peer_id,
            message="Тип повреждения обновлён.",
            keyboard=damage_edit_keyboard(request_id, active_damages),
            random_id=0,
        )
    except Exception as exc:
        logger.error("Failed to edit damage: {}", exc)
        await api.messages.send(peer_id=event.peer_id, message="Ошибка при изменении повреждения.", random_id=0)
