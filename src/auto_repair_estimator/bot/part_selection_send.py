from __future__ import annotations

from vkbottle import API

from auto_repair_estimator.bot.keyboards.part_selection import part_selection_keyboards_list

PART_SELECTION_CONTINUATION = "Остальные детали (продолжение):"


async def send_part_selection_messages(
    api: API,
    peer_id: int,
    request_id: str,
    first_message: str,
    continuation_message: str = PART_SELECTION_CONTINUATION,
) -> None:
    keybs = part_selection_keyboards_list(request_id)
    for i, kb in enumerate(keybs):
        text = first_message if i == 0 else continuation_message
        await api.messages.send(peer_id=peer_id, message=text, keyboard=kb, random_id=0)
