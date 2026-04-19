from __future__ import annotations

import httpx
from loguru import logger
from vkbottle import API
from vkbottle.bot import Message

from auto_repair_estimator.bot.backend_client import BackendClient


async def handle_photo(message: Message, backend: BackendClient, api: API) -> None:
    """Handle one or more photo attachments in a single VK message.

    VK allows up to 10 attachments per message. For each photo we create a
    separate ML ``RepairRequest`` so the user can receive per-photo results.
    """

    photos = message.get_photo_attachments()
    if not photos:
        return

    total = len(photos)
    if total > 1:
        await message.answer(f"Принято {total} фото. Обрабатываю каждое отдельной заявкой, подождите...")
    else:
        await message.answer("Загружаю фотографию, подождите...")

    peer_id = message.peer_id
    user_id = message.from_id

    success = 0
    for index, photo in enumerate(photos, start=1):
        try:
            await _process_single_photo(message, backend, photo, peer_id=peer_id, user_id=user_id, index=index)
            success += 1
        except Exception as exc:
            logger.error("Failed to process photo {}/{}: {}", index, total, exc)

    if success == 0:
        await message.answer("Не удалось обработать фотографии. Попробуйте ещё раз или используйте ручной ввод.")
        return

    if total == 1:
        await message.answer(
            "Фотография получена. Запрос обрабатывается (~15 секунд).\nЯ пришлю результат, когда будет готово."
        )
    else:
        await message.answer(
            f"Принято {success} из {total} фото. По каждому я пришлю отдельный результат (~15 секунд на фото)."
        )


async def _process_single_photo(
    message: Message,
    backend: BackendClient,
    photo: object,
    *,
    peer_id: int,
    user_id: int,
    index: int,
) -> None:
    data = await backend.create_request(chat_id=peer_id, user_id=user_id, mode="ml")
    request_id = data["id"]

    sizes = getattr(photo, "sizes", None)
    if not sizes:
        raise ValueError("Photo has no sizes")
    largest = max(sizes, key=lambda s: (getattr(s, "width", 0) or 0) * (getattr(s, "height", 0) or 0))
    photo_url = largest.url

    async with httpx.AsyncClient() as client:
        photo_resp = await client.get(photo_url)
        photo_resp.raise_for_status()
        image_bytes = photo_resp.content

    image_key = f"raw-images/{request_id}.jpg"

    presigned_url = data.get("presigned_put_url")
    if presigned_url:
        async with httpx.AsyncClient() as client:
            upload_resp = await client.put(
                presigned_url,
                content=image_bytes,
                headers={"Content-Type": "image/jpeg"},
            )
            upload_resp.raise_for_status()
    else:
        logger.warning("No presigned_put_url received for request_id={} (photo {})", request_id, index)

    await backend.upload_photo(request_id=request_id, image_key=image_key)
    logger.info("Uploaded photo {} for request_id={}", index, request_id)
