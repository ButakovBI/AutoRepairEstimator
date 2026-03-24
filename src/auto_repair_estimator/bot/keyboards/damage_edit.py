from __future__ import annotations

from typing import Any

from vkbottle import Callback, Keyboard, KeyboardButtonColor

from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


def inference_result_keyboard(request_id: str) -> str:
    kb = Keyboard(inline=True)
    kb.add(Callback("Подтвердить", payload={"cmd": "confirm", "rid": request_id}), color=KeyboardButtonColor.POSITIVE)
    kb.add(Callback("Подправить", payload={"cmd": "edit", "a": "start_edit", "rid": request_id, "did": ""}))
    return kb.get_json()


def damage_edit_keyboard(request_id: str, damages: list[dict[str, Any]]) -> str:
    kb = Keyboard(inline=True)
    for i, damage in enumerate(damages, 1):
        part_label = PART_LABELS.get(damage.get("part_type", ""), damage.get("part_type", ""))
        damage_label = DAMAGE_LABELS.get(damage.get("damage_type", ""), damage.get("damage_type", ""))
        damage_id = damage.get("id", "")
        kb.add(
            Callback(
                f"{i}: {part_label} — {damage_label}",
                payload={"cmd": "edit", "a": "edit_type", "rid": request_id, "did": damage_id},
            )
        )
        kb.add(
            Callback(
                f"Удалить {i}",
                payload={"cmd": "edit", "a": "delete", "rid": request_id, "did": damage_id},
            ),
            color=KeyboardButtonColor.NEGATIVE,
        )
        pair_in_row = i % 2 == 0 or i == len(damages)
        if pair_in_row:
            kb.row()
    kb.add(Callback("Добавить повреждение", payload={"cmd": "addmore", "rid": request_id}))
    kb.add(
        Callback("Готово", payload={"cmd": "confirm", "rid": request_id}),
        color=KeyboardButtonColor.POSITIVE,
    )
    return kb.get_json()


def edit_damage_type_keyboard(request_id: str, damage_id: str) -> str:
    kb = Keyboard(inline=True)
    items = list(DAMAGE_LABELS.items())
    for i, (damage_type, label) in enumerate(items):
        kb.add(Callback(label, payload={"cmd": "edtype", "rid": request_id, "did": damage_id, "dt": damage_type}))
        if i % 2 == 1 and i < len(items) - 1:
            kb.row()
    return kb.get_json()


def add_more_or_confirm_keyboard(request_id: str) -> str:
    kb = Keyboard(inline=True)
    kb.add(Callback("Добавить ещё", payload={"cmd": "addmore", "rid": request_id}))
    kb.add(
        Callback("Подтвердить", payload={"cmd": "confirm", "rid": request_id}),
        color=KeyboardButtonColor.POSITIVE,
    )
    return kb.get_json()
