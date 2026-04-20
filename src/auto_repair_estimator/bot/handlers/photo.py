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

    # VK's conversation_message_id is the most stable per-chat message
    # identifier; combined with peer_id it forms a dedup key that survives
    # Long Poll retransmissions without pulling in the full message_id
    # (which changes on some edits).
    base_message_id = (
        getattr(message, "conversation_message_id", None)
        or getattr(message, "id", None)
        or 0
    )

    success = 0
    for index, photo in enumerate(photos, start=1):
        # Per-photo key — if the user sends the same N photos twice, each
        # attachment position deduplicates independently.
        idem_key = f"{peer_id}:{base_message_id}:{index}"
        try:
            await _process_single_photo(
                message,
                backend,
                photo,
                peer_id=peer_id,
                user_id=user_id,
                index=index,
                idempotency_key=idem_key,
            )
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
    idempotency_key: str | None = None,
) -> None:
    data = await backend.create_request(
        chat_id=peer_id,
        user_id=user_id,
        mode="ml",
        idempotency_key=idempotency_key,
    )
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
    if not presigned_url:
        # Without a presigned URL we cannot put the bytes into storage,
        # which means ``upload_photo`` would enqueue an inference that
        # can't possibly run (the worker would 404 on the image key) and
        # leave the request stuck in QUEUED until the watchdog kills it.
        # Better to surface a clean error to the user immediately than
        # to corrupt the pipeline with half-uploaded requests.
        logger.error(
            "No presigned_put_url for request_id={} (photo {}); aborting upload",
            request_id,
            index,
        )
        raise RuntimeError("backend did not return a presigned_put_url")

    async with httpx.AsyncClient() as client:
        upload_resp = await client.put(
            presigned_url,
            content=image_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        upload_resp.raise_for_status()

    await backend.upload_photo(request_id=request_id, image_key=image_key)
    logger.info("Uploaded photo {} for request_id={}", index, request_id)
