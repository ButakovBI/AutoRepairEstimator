"""Damage-type inline keyboard, filtered by ``part_type``.

Before this filter existed the bot rendered all eight DamageType values for
every part, letting the user pick physically impossible pairs
(``scratch`` on a headlight, ``flat_tire`` on a door). The single source
of truth for which combinations are meaningful is
:data:`auto_repair_estimator.backend.domain.value_objects.part_damage_compatibility.PART_DAMAGE_COMPATIBILITY`;
this module just projects that onto a VK inline keyboard.
"""

from __future__ import annotations

from vkbottle import Callback, Keyboard, KeyboardButtonColor

from auto_repair_estimator.backend.domain.value_objects.part_damage_compatibility import (
    PART_DAMAGE_COMPATIBILITY,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageType,
    PartType,
)
from auto_repair_estimator.bot.labels import DAMAGE_LABELS


def _allowed_damage_values(part_type: str) -> list[str]:
    """Resolve ``part_type`` to the ordered list of damage enum values the UI
    should expose.

    Unknown ``part_type`` values (legacy keyboards, callback-data injection)
    degrade to an empty list rather than raising — the caller (backend
    validation) will return a 400 anyway, and emitting a keyboard with zero
    buttons is safer than crashing the handler.
    """

    try:
        part_enum = PartType(part_type)
    except ValueError:
        return []

    allowed = PART_DAMAGE_COMPATIBILITY.get(part_enum, frozenset())
    # Preserve DamageType declaration order so the UI stays deterministic
    # across renders (VK re-renders keyboards in whatever order we send).
    return [dt.value for dt in DamageType if dt in allowed]


def damage_type_selection_keyboard(request_id: str, part_type: str) -> str:
    kb = Keyboard(inline=True)
    allowed_values = _allowed_damage_values(part_type)
    for i, damage_type in enumerate(allowed_values):
        label = DAMAGE_LABELS.get(damage_type, damage_type)
        kb.add(
            Callback(
                label,
                payload={"cmd": "dmg", "rid": request_id, "pt": part_type, "dt": damage_type},
            )
        )
        if i % 2 == 1 and i < len(allowed_values) - 1:
            kb.row()
    # Back-to-parts is the one navigation affordance that is always safe
    # here: picking a damage type is the only action on this screen, so
    # "go back" unambiguously means "re-pick a different part". Bug
    # report #3 asked for a Back button "there where it's required", and
    # this is that screen.
    kb.row()
    kb.add(
        Callback("← К выбору детали", payload={"cmd": "back_parts", "rid": request_id}),
        color=KeyboardButtonColor.SECONDARY,
    )
    return str(kb.get_json())
