from __future__ import annotations

import httpx
from loguru import logger
from vkbottle import API
from vkbottle.bot import Message

from auto_repair_estimator.bot.backend_client import BackendClient


async def handle_photo(message: Message, backend: BackendClient, api: API) -> None:
    photos = message.get_photo_attachments()
    if not photos:
        return

    await message.answer("Загружаю фотографию, подождите...")

    peer_id = message.peer_id
    user_id = message.from_id

    try:
        data = await backend.create_request(chat_id=peer_id, user_id=user_id, mode="ml")
        request_id = data["id"]

        photo = photos[0]
        if not photo.sizes:
            raise ValueError("Photo has no sizes")

        largest = max(photo.sizes, key=lambda s: (s.width or 0) * (s.height or 0))
        photo_url = largest.url

        async with httpx.AsyncClient() as client:
            photo_resp = await client.get(photo_url)
            photo_resp.raise_for_status()
            image_data = photo_resp.content

        image_key = f"raw-images/{request_id}.jpg"

        presigned_url = data.get("presigned_put_url")
        if presigned_url:
            async with httpx.AsyncClient() as client:
                upload_resp = await client.put(
                    presigned_url,
                    content=image_data,
                    headers={"Content-Type": "image/jpeg"},
                )
                upload_resp.raise_for_status()
        else:
            logger.warning("No presigned_put_url received for request_id={}", request_id)

        await backend.upload_photo(request_id=request_id, image_key=image_key)
        await message.answer(
            "Фотография получена! Запрос обрабатывается (~15 секунд).\nЯ пришлю результат, когда будет готово."
        )
    except Exception as exc:
        logger.error("Failed to process photo: {}", exc)
        await message.answer("Не удалось обработать фотографию. Попробуйте ещё раз или используйте ручной ввод.")
