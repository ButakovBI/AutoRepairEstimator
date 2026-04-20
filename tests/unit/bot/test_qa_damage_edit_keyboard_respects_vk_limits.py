"""QA: damage_edit_keyboard must respect VK inline-keyboard limits.

VK's documentation caps inline keyboards at:

* ``VK_INLINE_MAX_BUTTONS`` = 10 buttons total
* ``VK_INLINE_MAX_ROWS`` = 6 rows

If either is exceeded, the VK API returns ``VKAPIError_911`` and the bot
silently fails to send the keyboard — leaving the user stuck on the
previous message with no way to edit or confirm their damages.

The current ``damage_edit_keyboard`` implementation emits **two** buttons
per damage (edit + delete) plus **two** fixed trailing buttons
("Добавить повреждение" and "Готово"). The plan itself anticipates up to
5 simultaneously detected damages. With 5 damages the keyboard therefore
contains 5 * 2 + 2 = 12 buttons, exceeding the 10-button cap.

This test exercises a range of damage counts and asserts that neither
limit is violated, by parsing the JSON payload that the keyboard
factory returns.
"""

from __future__ import annotations

import json

import pytest

from auto_repair_estimator.bot.keyboards.damage_edit import damage_edit_keyboard
from auto_repair_estimator.bot.vk_limits import VK_INLINE_MAX_BUTTONS, VK_INLINE_MAX_ROWS
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


def _build_damages(n: int) -> list[dict[str, object]]:
    """Build ``n`` plausible damages using real part/damage type values so
    that the label lookup in ``damage_edit_keyboard`` behaves as in prod."""

    return [
        {
            "id": f"dmg-{i}",
            "part_type": PartType.HOOD.value,
            "damage_type": DamageType.SCRATCH.value,
        }
        for i in range(n)
    ]


def _count_buttons_and_rows(keyboard_json: str) -> tuple[int, int]:
    payload = json.loads(keyboard_json)
    buttons = payload.get("buttons", [])
    rows = len(buttons)
    total = sum(len(row) for row in buttons)
    return total, rows


# The plan anticipates up to ~5 damages per request (§"ML-режим" breakdown).
# We also test the boundary at 4 to lock in the exact failing point.
@pytest.mark.parametrize("damage_count", [3, 4, 5, 6])
def test_damage_edit_keyboard_does_not_exceed_vk_button_cap(damage_count: int) -> None:
    """With ``N`` damages the keyboard emits ``2*N + 2`` buttons today; any
    realistic repair session with ≥5 damages trips VK's 10-button cap."""

    # Arrange
    damages = _build_damages(damage_count)

    # Act
    keyboard_json = damage_edit_keyboard(request_id="r-1", damages=damages)
    total_buttons, total_rows = _count_buttons_and_rows(keyboard_json)

    # Assert — button cap
    assert total_buttons <= VK_INLINE_MAX_BUTTONS, (
        f"damage_edit_keyboard emitted {total_buttons} buttons for {damage_count} damages, "
        f"exceeding the VK inline-keyboard cap of {VK_INLINE_MAX_BUTTONS}. "
        "VK will reject the message (VKAPIError_911) and the user will be stuck. "
        "Paginate damages or collapse edit+delete into a single 'manage' button."
    )
    # Assert — row cap
    assert total_rows <= VK_INLINE_MAX_ROWS, (
        f"damage_edit_keyboard emitted {total_rows} rows for {damage_count} damages, "
        f"exceeding the VK inline-keyboard cap of {VK_INLINE_MAX_ROWS}."
    )
