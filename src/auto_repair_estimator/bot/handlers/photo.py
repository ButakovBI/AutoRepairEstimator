from __future__ import annotations

import httpx
from loguru import logger
from vkbottle import API
from vkbottle.bot import Message

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.session_lifecycle import (
    PREVIOUS_REQUEST_ABANDONED_NOTICE,
    abandon_active_session,
)

MULTIPLE_PHOTOS_NOTICE = (
    "За один раз можно обработать только одну фотографию — беру первую. "
    "Если хотите проанализировать и остальные, отправьте их по одной "
    "после получения результата."
)


async def handle_photo(message: Message, backend: BackendClient, api: API) -> None:
    """Handle photo attachments, enforcing the single-active-session invariant.

    Two invariants matter here:

    * **One active session per chat.** If the chat has a non-terminal
      :class:`RepairRequest` (ML mid-flight, manual session with half-
      added damages, …), we abandon it *before* starting the new ML
      one. The user is told what happened through the shared
      ``PREVIOUS_REQUEST_ABANDONED_NOTICE`` prefix so nobody loses
      progress silently.
    * **One photo per message.** VK lets a single message carry up to
      10 attachments. Processing them all used to create 10 parallel
      ML requests — a fan-out that directly violates the first
      invariant and made the bot unusable if the user photoed a car
      from every angle and sent them at once. We now process only the
      first photo and tell the user the others were skipped.
    """

    photos = message.get_photo_attachments()
    if not photos:
        return

    peer_id = message.peer_id
    user_id = message.from_id
    total = len(photos)

    intro_parts: list[str] = []
    abandoned = await abandon_active_session(backend, peer_id)
    if abandoned is not None:
        intro_parts.append(PREVIOUS_REQUEST_ABANDONED_NOTICE)
    if total > 1:
        intro_parts.append(MULTIPLE_PHOTOS_NOTICE)
    intro_parts.append("Загружаю фотографию, подождите...")
    await message.answer("\n\n".join(intro_parts))

    # VK's conversation_message_id is the most stable per-chat message
    # identifier; combined with peer_id it forms a dedup key that survives
    # Long Poll retransmissions without pulling in the full message_id
    # (which changes on some edits).
    base_message_id = (
        getattr(message, "conversation_message_id", None)
        or getattr(message, "id", None)
        or 0
    )
    # Index "1" is baked into the key so VK redeliveries of the exact
    # same message collapse to the same RepairRequest via idempotency,
    # while a genuinely new photo in a new VK message gets a fresh one.
    idem_key = f"{peer_id}:{base_message_id}:1"

    first_photo = photos[0]
    try:
        await _process_single_photo(
            message,
            backend,
            first_photo,
            peer_id=peer_id,
            user_id=user_id,
            index=1,
            idempotency_key=idem_key,
        )
    except Exception as exc:
        logger.error("Failed to process photo: {}", exc)
        await message.answer(
            "Не удалось обработать фотографию. Попробуйте ещё раз "
            "или используйте ручной ввод."
        )
        return

    await message.answer(
        "Фотография получена. Запрос обрабатывается (~15 секунд).\n"
        "Я пришлю результат, когда будет готово."
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
