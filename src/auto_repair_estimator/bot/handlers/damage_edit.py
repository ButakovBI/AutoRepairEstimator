from __future__ import annotations

from typing import Any

from loguru import logger
from vkbottle import API, PhotoMessageUploader
from vkbottle.bot import MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.damage_grouping import group_damages
from auto_repair_estimator.bot.damage_list_format import format_damage_list
from auto_repair_estimator.bot.keyboards.damage_edit import (
    damage_edit_keyboards_list,
    edit_damage_type_keyboard,
    group_retype_keyboard,
    group_submenu_keyboard,
    inference_result_keyboard,
)
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS
from auto_repair_estimator.bot.part_selection_send import send_part_selection_messages

EDIT_HEADER = "Редактирование повреждений:"
# Continuation header for pages beyond the first in the paginated edit
# list. Keeping it terse keeps the eyes on the buttons, which is the
# actionable content of the message.
EDIT_HEADER_CONTINUATION = "Ещё повреждения:"


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


async def _send_edit_screen(
    api: API,
    peer_id: int,
    request_id: str,
    damages: list[dict[str, Any]],
    *,
    first_header: str = EDIT_HEADER,
) -> None:
    """Render the paginated edit list.

    Sends the damage-list text followed by ``N`` keyboard-carrying messages
    (one per VK inline-keyboard page). The text is sent once as a
    standalone message so the user can always see the full numbered list
    of what's in the basket, regardless of how many keyboard pages the
    groups fan out into. This replaces the old single-keyboard path that
    silently dropped damages past a hard cap.
    """

    active = [d for d in damages if not d.get("is_deleted", False)]
    if active:
        body = format_damage_list(damages, header=first_header)
    else:
        # Preserve the action header (e.g. "Повреждение удалено.") even
        # when the basket ends up empty — dropping it to the default
        # "пока нет" copy loses the "action just happened" signal.
        body = f"{first_header}\n\nПока нет добавленных повреждений."
    keyboards = damage_edit_keyboards_list(request_id, damages)
    # Attach the first keyboard to the list message so users on the last
    # page can tap immediately without an empty "header-only" scroll.
    for index, kb in enumerate(keyboards):
        text = body if index == 0 else EDIT_HEADER_CONTINUATION
        await api.messages.send(peer_id=peer_id, message=text, keyboard=kb, random_id=0)


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
        await _send_edit_screen(api, event.peer_id, str(request_id), damages)

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
        except Exception as exc:
            logger.error("Failed to delete damage: {}", exc)
            await api.messages.send(peer_id=event.peer_id, message="Ошибка при удалении повреждения.", random_id=0)
            return
        await _send_edit_screen(
            api,
            event.peer_id,
            str(request_id),
            active_damages,
            first_header="Повреждение удалено.\n\nТекущий список:",
        )


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
    except Exception as exc:
        logger.error("Failed to edit damage: {}", exc)
        await api.messages.send(peer_id=event.peer_id, message="Ошибка при изменении повреждения.", random_id=0)
        return
    await _send_edit_screen(
        api,
        event.peer_id,
        str(request_id),
        active_damages,
        first_header="Тип повреждения обновлён.\n\nТекущий список:",
    )


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
    await _send_edit_screen(api, event.peer_id, rid, active_damages)


async def handle_group_action(
    event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API
) -> None:
    """Entry point for the ``grp`` callback family (grouped damage edits).

    Dispatches the five group actions wired from ``damage_edit`` and
    ``group_submenu`` keyboards. All actions operate on the current
    ``(part_type, damage_type)`` bucket resolved against the freshest
    backend snapshot — stale ids from a prior delete never leak into the
    next round-trip.
    """

    action = payload.get("a")
    request_id = payload.get("rid")
    part_type = payload.get("pt")
    damage_type_old = payload.get("dt")
    if not action or not request_id or not part_type or not damage_type_old:
        logger.warning("handle_group_action received malformed payload: {}", payload)
        await api.messages.send(
            peer_id=event.peer_id,
            message="Некорректная кнопка. Нажмите «Начать», чтобы начать заново.",
            random_id=0,
        )
        return
    rid = str(request_id)

    if action == "open":
        count = await _count_group_members(backend, rid, str(part_type), str(damage_type_old))
        if count == 0:
            await _send_group_vanished_and_refresh(api, event.peer_id, rid, backend)
            return
        part_label = PART_LABELS.get(str(part_type), str(part_type))
        damage_label = DAMAGE_LABELS.get(str(damage_type_old), str(damage_type_old))
        await api.messages.send(
            peer_id=event.peer_id,
            message=f"Группа: {part_label} — {damage_label} (×{count})\n\nВыберите действие:",
            keyboard=group_submenu_keyboard(rid, str(part_type), str(damage_type_old), count),
            random_id=0,
        )
        return

    if action == "retype":
        await api.messages.send(
            peer_id=event.peer_id,
            message="Выберите новый тип повреждения (применится ко всей группе):",
            keyboard=group_retype_keyboard(rid, str(part_type), str(damage_type_old)),
            random_id=0,
        )
        return

    if action == "apply_retype":
        new_damage_type = payload.get("nd")
        if not new_damage_type:
            logger.warning("apply_retype missing nd: {}", payload)
            await api.messages.send(
                peer_id=event.peer_id,
                message="Некорректная кнопка. Нажмите «Начать», чтобы начать заново.",
                random_id=0,
            )
            return
        damage_ids = await _resolve_group_ids(backend, rid, str(part_type), str(damage_type_old))
        if not damage_ids:
            await _send_group_vanished_and_refresh(api, event.peer_id, rid, backend)
            return
        failures = 0
        for did in damage_ids:
            try:
                await backend.edit_damage(rid, did, str(new_damage_type))
            except Exception as exc:
                failures += 1
                logger.error("Bulk retype failed for rid={} did={}: {}", rid, did, exc)
        if failures == len(damage_ids):
            await api.messages.send(
                peer_id=event.peer_id,
                message="Ошибка при изменении повреждений. Попробуйте ещё раз.",
                random_id=0,
            )
            return
        # Partial failures are communicated as a soft notice; the rest of
        # the edit list is still rendered because the user almost always
        # wants to continue editing, not start from scratch.
        header = (
            "Тип повреждений обновлён."
            if failures == 0
            else f"Тип обновлён частично ({len(damage_ids) - failures}/{len(damage_ids)}). Некоторые повреждения не удалось обновить."
        )
        await _refetch_and_send_edit_screen(api, event.peer_id, rid, backend, header=f"{header}\n\nТекущий список:")
        return

    if action == "del_all":
        damage_ids = await _resolve_group_ids(backend, rid, str(part_type), str(damage_type_old))
        if not damage_ids:
            await _send_group_vanished_and_refresh(api, event.peer_id, rid, backend)
            return
        failures = 0
        for did in damage_ids:
            try:
                await backend.delete_damage(rid, did)
            except Exception as exc:
                failures += 1
                logger.error("Bulk delete failed for rid={} did={}: {}", rid, did, exc)
        if failures == len(damage_ids):
            await api.messages.send(
                peer_id=event.peer_id,
                message="Ошибка при удалении повреждений. Попробуйте ещё раз.",
                random_id=0,
            )
            return
        header = (
            "Группа удалена."
            if failures == 0
            else f"Удалено частично ({len(damage_ids) - failures}/{len(damage_ids)}). Некоторые повреждения не удалось удалить."
        )
        await _refetch_and_send_edit_screen(api, event.peer_id, rid, backend, header=f"{header}\n\nТекущий список:")
        return

    if action == "del_one":
        damage_ids = await _resolve_group_ids(backend, rid, str(part_type), str(damage_type_old))
        if not damage_ids:
            await _send_group_vanished_and_refresh(api, event.peer_id, rid, backend)
            return
        try:
            await backend.delete_damage(rid, damage_ids[0])
        except Exception as exc:
            logger.error("del_one failed for rid={} did={}: {}", rid, damage_ids[0], exc)
            await api.messages.send(
                peer_id=event.peer_id,
                message="Ошибка при удалении повреждения. Попробуйте ещё раз.",
                random_id=0,
            )
            return
        await _refetch_and_send_edit_screen(
            api, event.peer_id, rid, backend, header="Одно повреждение удалено.\n\nТекущий список:"
        )
        return

    logger.warning("handle_group_action received unknown action: {}", action)
    await api.messages.send(
        peer_id=event.peer_id,
        message="Некорректная кнопка. Нажмите «Начать», чтобы начать заново.",
        random_id=0,
    )


async def _count_group_members(
    backend: BackendClient, request_id: str, part_type: str, damage_type: str
) -> int:
    ids = await _resolve_group_ids(backend, request_id, part_type, damage_type)
    return len(ids)


async def _resolve_group_ids(
    backend: BackendClient, request_id: str, part_type: str, damage_type: str
) -> list[str]:
    """Return the damage ids for a ``(part, damage_type)`` group as of now.

    Always fetches fresh: ``(rid, pt, dt)`` is the only group identity we
    encode in the payload, so each call must re-resolve it against the
    latest backend state to avoid acting on stale ids.
    """

    try:
        data = await backend.get_request(request_id)
    except Exception as exc:
        logger.warning("Could not refetch for group resolve rid={}: {}", request_id, exc)
        return []
    damages = data.get("damages", []) if isinstance(data, dict) else []
    for group in group_damages(damages):
        if group.part_type == part_type and group.damage_type == damage_type:
            return list(group.damage_ids)
    return []


async def _refetch_and_send_edit_screen(
    api: API,
    peer_id: int,
    request_id: str,
    backend: BackendClient,
    *,
    header: str,
) -> None:
    try:
        data = await backend.get_request(request_id)
    except Exception as exc:
        logger.error("Could not refetch after bulk op rid={}: {}", request_id, exc)
        await api.messages.send(
            peer_id=peer_id,
            message="Действие выполнено, но не удалось обновить список. Нажмите «Подправить», чтобы перечитать данные.",
            random_id=0,
        )
        return
    active = [d for d in data.get("damages", []) if not d.get("is_deleted", False)]
    await _send_edit_screen(api, peer_id, request_id, active, first_header=header)


async def _send_group_vanished_and_refresh(
    api: API, peer_id: int, request_id: str, backend: BackendClient
) -> None:
    """Fallback for when a group's members are gone by the time we look.

    Possible whenever a user pressed two group buttons in quick
    succession (e.g. "Удалить все" twice) and the second tap races the
    first delete. Refreshing the edit screen tells the user the
    operation they were about to repeat is already done.
    """

    await api.messages.send(
        peer_id=peer_id,
        message="Этой группы повреждений больше нет. Обновляю список.",
        random_id=0,
    )
    await _refetch_and_send_edit_screen(api, peer_id, request_id, backend, header=EDIT_HEADER)


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
