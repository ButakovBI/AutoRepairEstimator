from __future__ import annotations

from typing import Any

from loguru import logger
from vkbottle import API
from vkbottle.bot import MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.keyboards.start import start_keyboard
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


def _format_range(
    min_value: float,
    max_value: float,
    *,
    suffix: str,
    fmt: str = ",.0f",
) -> str:
    """Render a ``[min, max]`` pair as ``"N suffix"`` or ``"N–M suffix"``.

    Using an en-dash (U+2013) to match the thesis tables exactly.
    """

    if min_value == max_value:
        return f"{min_value:{fmt}} {suffix}"
    return f"{min_value:{fmt}}\u2013{max_value:{fmt}} {suffix}"


def _format_hours(min_h: float, max_h: float) -> str:
    """Hours are rendered with 1 decimal place when they are fractional
    (e.g. 0.5 hours of polishing), otherwise as whole numbers."""
    needs_decimal = (min_h % 1) or (max_h % 1)
    fmt = ",.1f" if needs_decimal else ",.0f"
    return _format_range(min_h, max_h, suffix="ч", fmt=fmt)


async def handle_confirm(event: MessageEvent, payload: dict[str, Any], backend: BackendClient, api: API) -> None:
    request_id = payload.get("rid")
    # ``rid`` is expected to be a string UUID but we accept any truthy value
    # defensively — if the backend rejects the shape, the generic error
    # branch below will report it to the user instead of crashing the
    # handler. Only None / empty string count as a malformed payload.
    if not request_id:
        logger.warning("handle_confirm received malformed payload: {}", payload)
        await api.messages.send(
            peer_id=event.peer_id,
            message="Ошибка: некорректная кнопка. Напишите /start, чтобы начать заново.",
            random_id=0,
        )
        return

    try:
        result = await backend.confirm_pricing(request_id)
    except Exception as exc:
        logger.error("Failed to confirm pricing: {}", exc)
        await api.messages.send(
            peer_id=event.peer_id, message="Ошибка при расчёте стоимости. Попробуйте ещё раз.", random_id=0
        )
        return

    total_cost_min = float(result.get("total_cost_min", 0.0))
    total_cost_max = float(result.get("total_cost_max", 0.0))
    total_hours_min = float(result.get("total_hours_min", 0.0))
    total_hours_max = float(result.get("total_hours_max", 0.0))
    breakdown: list[dict[str, Any]] = result.get("breakdown", [])
    notes: list[str] = result.get("notes", [])

    lines: list[str] = []
    if breakdown:
        lines.append(
            f"Стоимость ремонта: {_format_range(total_cost_min, total_cost_max, suffix='руб.')}"
        )
        lines.append(
            f"Приблизительное время: {_format_hours(total_hours_min, total_hours_max)}"
        )
        lines.append("")
        lines.append("Детализация:")
        for item in breakdown:
            part_label = PART_LABELS.get(item.get("part_type", ""), item.get("part_type", ""))
            damage_label = DAMAGE_LABELS.get(item.get("damage_type", ""), item.get("damage_type", ""))
            cost = _format_range(
                float(item.get("cost_min", 0.0)),
                float(item.get("cost_max", 0.0)),
                suffix="руб.",
            )
            hours = _format_hours(
                float(item.get("hours_min", 0.0)),
                float(item.get("hours_max", 0.0)),
            )
            # The treatment suffix tells the user why a price is as high
            # as it is. "Замена" is visually jarring in a range like
            # "18 000 руб. за царапину", so we surface it inline for
            # replacement-class damages only. The backend picks the
            # treatment via ``causes_replacement`` — see
            # ``PricingService.calculate``. Anything non-replacement
            # falls through without a suffix: the price range is
            # self-explanatory for painting / dent work.
            treatment_suffix = (
                " — замена" if item.get("treatment") == "replacement" else ""
            )
            lines.append(
                f"  - {part_label} — {damage_label}{treatment_suffix}: {cost} ({hours})"
            )
    else:
        # All damages were routed to a tyre shop / had no rule: just show notes.
        lines.append("Кузовной ремонт по этой заявке не требуется.")

    for note in notes:
        lines.append("")
        lines.append(note)

    lines.append("")
    lines.append("Чтобы оценить ещё одно повреждение, нажмите «Начать».")
    # Attach the start-keyboard so the user has a one-tap escape to a new
    # scenario — the previous "type /start" copy worked but forced the
    # user to type, which is friction in VK chat (bug #1 in the UX
    # round). The message is terminal (request is now DONE), so this is
    # the only actionable affordance that still makes sense here.
    await api.messages.send(
        peer_id=event.peer_id,
        message="\n".join(lines),
        keyboard=start_keyboard(),
        random_id=0,
    )
