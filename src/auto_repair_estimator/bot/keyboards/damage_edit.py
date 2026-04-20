from __future__ import annotations

from typing import Any

from vkbottle import Callback, Keyboard, KeyboardButtonColor

from auto_repair_estimator.backend.domain.value_objects.part_damage_compatibility import (
    PART_DAMAGE_COMPATIBILITY,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageType,
    PartType,
)
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


def inference_result_keyboard(request_id: str) -> str:
    kb = Keyboard(inline=True)
    kb.add(Callback("Подтвердить", payload={"cmd": "confirm", "rid": request_id}), color=KeyboardButtonColor.POSITIVE)
    kb.add(Callback("Подправить", payload={"cmd": "edit", "a": "start_edit", "rid": request_id, "did": ""}))
    return kb.get_json()


# VK caps each inline keyboard at 10 buttons and 6 rows. We reserve two
# buttons for the trailing "Add / Confirm" controls, which leaves at most
# 8 per-damage buttons. Packing two damages per row keeps us under the
# 6-row cap for up to 8 simultaneous damages — beyond that we drop the
# tail and nudge the user to confirm or remove extras, which is vastly
# better UX than sending a keyboard VK silently refuses.
_DAMAGE_ROWS_AVAILABLE = 4
_DAMAGES_PER_ROW = 2
_MAX_DAMAGES_IN_EDIT_KB = _DAMAGE_ROWS_AVAILABLE * _DAMAGES_PER_ROW  # 8


def damage_edit_keyboard(request_id: str, damages: list[dict[str, Any]]) -> str:
    kb = Keyboard(inline=True)
    visible = damages[:_MAX_DAMAGES_IN_EDIT_KB]

    for i, damage in enumerate(visible, 1):
        part_type_value = damage.get("part_type", "")
        part_label = PART_LABELS.get(part_type_value, part_type_value)
        damage_label = DAMAGE_LABELS.get(damage.get("damage_type", ""), damage.get("damage_type", ""))
        damage_id = damage.get("id", "")
        # One collapsed "manage" button per damage: tapping opens a sub-menu
        # with damage-type choices and a Delete button. The sub-menu must
        # filter damage types by the current part (headlight -> only
        # broken_headlight, etc.), so we propagate ``pt`` through the
        # callback payload — otherwise the sub-handler would have to refetch
        # the damage from the backend just to know the part.
        kb.add(
            Callback(
                f"{i}. {part_label} — {damage_label}",
                payload={
                    "cmd": "edit",
                    "a": "edit_type",
                    "rid": request_id,
                    "did": damage_id,
                    "pt": part_type_value,
                },
            )
        )
        if i % _DAMAGES_PER_ROW == 0 and i != len(visible):
            kb.row()

    kb.row()
    kb.add(Callback("Добавить повреждение", payload={"cmd": "addmore", "rid": request_id}))
    kb.add(
        Callback("Готово", payload={"cmd": "confirm", "rid": request_id}),
        color=KeyboardButtonColor.POSITIVE,
    )
    return kb.get_json()


def _allowed_damage_values_for(part_type: str) -> list[str]:
    """Same filter logic as in ``damage_type_selection`` but local to avoid
    a cross-module import cycle (both modules import DAMAGE_LABELS).
    Unknown parts degrade to the full list so an editor can still change
    the type on a legacy damage whose part somehow falls outside the enum.
    """
    try:
        part_enum = PartType(part_type)
    except ValueError:
        return [dt.value for dt in DamageType]
    allowed = PART_DAMAGE_COMPATIBILITY.get(part_enum, frozenset())
    return [dt.value for dt in DamageType if dt in allowed] or [dt.value for dt in DamageType]


def edit_damage_type_keyboard(request_id: str, damage_id: str, part_type: str = "") -> str:
    kb = Keyboard(inline=True)
    # part_type can be empty for legacy callbacks that predate the pt-in-payload
    # change; in that case we degrade to showing the full list rather than an
    # empty keyboard (the backend will reject an incompatible edit anyway).
    allowed_values = _allowed_damage_values_for(part_type)
    for i, damage_type in enumerate(allowed_values):
        label = DAMAGE_LABELS.get(damage_type, damage_type)
        kb.add(Callback(label, payload={"cmd": "edtype", "rid": request_id, "did": damage_id, "dt": damage_type}))
        if i % 2 == 1 and i < len(allowed_values) - 1:
            kb.row()
    # Delete lives on this sub-keyboard — the main damage_edit_keyboard no
    # longer has a per-damage Delete button (it would push the keyboard past
    # the VK 10-button cap for 5+ damages).
    kb.row()
    kb.add(
        Callback(
            "Удалить повреждение",
            payload={"cmd": "edit", "a": "delete", "rid": request_id, "did": damage_id},
        ),
        color=KeyboardButtonColor.NEGATIVE,
    )
    # "Back to the full list" escape hatch (bug #5): without it the user
    # would have to scroll up to the previous message to re-see all the
    # damages detected by ML, which is exactly what they complained about.
    kb.row()
    kb.add(
        Callback("← К списку повреждений", payload={"cmd": "back_edit", "rid": request_id}),
        color=KeyboardButtonColor.SECONDARY,
    )
    return kb.get_json()


def add_more_or_confirm_keyboard(request_id: str) -> str:
    kb = Keyboard(inline=True)
    kb.add(Callback("Добавить ещё", payload={"cmd": "addmore", "rid": request_id}))
    kb.add(
        Callback("Подтвердить", payload={"cmd": "confirm", "rid": request_id}),
        color=KeyboardButtonColor.POSITIVE,
    )
    return kb.get_json()
