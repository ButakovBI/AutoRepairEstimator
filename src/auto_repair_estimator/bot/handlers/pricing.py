from __future__ import annotations

from typing import Any

from loguru import logger
from vkbottle import API
from vkbottle.bot import MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


async def handle_confirm(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    request_id = payload["rid"]

    try:
        result = await backend.confirm_pricing(request_id)
    except Exception as exc:
        logger.error("Failed to confirm pricing: {}", exc)
        await api.messages.send(
            peer_id=event.peer_id, message="Ошибка при расчёте стоимости. Попробуйте ещё раз.", random_id=0
        )
        return

    total_cost = result.get("total_cost", 0.0)
    total_hours = result.get("total_hours", 0.0)
    breakdown = result.get("breakdown", [])

    lines = [f"Стоимость ремонта: {total_cost:,.0f} руб.", f"Приблизительное время: {total_hours:.1f} ч", ""]
    if breakdown:
        lines.append("Детализация:")
        for item in breakdown:
            part_label = PART_LABELS.get(item.get("part_type", ""), item.get("part_type", ""))
            damage_label = DAMAGE_LABELS.get(item.get("damage_type", ""), item.get("damage_type", ""))
            lines.append(
                f"  - {part_label} — {damage_label}: {item.get('cost', 0):,.0f} руб. ({item.get('hours', 0):.1f} ч)"
            )

    lines.append("\nДля нового запроса напишите /start или 'Начать'")
    await api.messages.send(peer_id=event.peer_id, message="\n".join(lines), random_id=0)
