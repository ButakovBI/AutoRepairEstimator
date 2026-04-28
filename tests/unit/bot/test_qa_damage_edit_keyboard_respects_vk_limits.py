"""QA: the paginated edit keyboards respect VK inline-keyboard limits.

VK's documentation caps inline keyboards at:

* ``VK_INLINE_MAX_BUTTONS`` = 10 buttons total
* ``VK_INLINE_MAX_ROWS`` = 6 rows

If either is exceeded the VK API returns ``VKAPIError_911`` and the bot
silently fails to send the keyboard — leaving the user stuck on the
previous message with no way to edit or confirm their damages. The old
``damage_edit_keyboard`` silently dropped damages past a hard 8-item
cap; the replacement ``damage_edit_keyboards_list`` paginates AND groups
instead, so both the limit and the "no dropped data" invariant must be
verified here.
"""

from __future__ import annotations

import json

import pytest

from auto_repair_estimator.bot.keyboards.damage_edit import damage_edit_keyboards_list
from auto_repair_estimator.bot.vk_limits import VK_INLINE_MAX_BUTTONS, VK_INLINE_MAX_ROWS
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


def _build_identical_damages(n: int) -> list[dict[str, object]]:
    """Build ``n`` damages that all share the same (part, damage_type).

    Identical pairs collapse into a single group button — this exercises
    the grouping branch that shields the UI from the "17 scratches on a
    door" overflow reported in the bug.
    """

    return [
        {
            "id": f"dmg-{i}",
            "part_type": PartType.DOOR.value,
            "damage_type": DamageType.SCRATCH.value,
        }
        for i in range(n)
    ]


def _build_distinct_damages(n: int) -> list[dict[str, object]]:
    """Build ``n`` damages with distinct damage_types so grouping cannot
    collapse them — this is the worst case for pagination."""

    # We cycle a fake damage_type suffix so every pair is unique. The
    # label lookup degrades gracefully for unknown values.
    return [
        {
            "id": f"dmg-{i}",
            "part_type": PartType.DOOR.value,
            "damage_type": f"synthetic_{i}",
        }
        for i in range(n)
    ]


def _count_buttons_and_rows(keyboard_json: str) -> tuple[int, int]:
    payload = json.loads(keyboard_json)
    buttons = payload.get("buttons", [])
    rows = len(buttons)
    total = sum(len(row) for row in buttons)
    return total, rows


@pytest.mark.parametrize("damage_count", [1, 5, 17, 50])
def test_identical_damages_collapse_to_one_group_button(damage_count: int) -> None:
    """Regression fence: 17 identical damages used to overflow the single-
    keyboard layout. The grouping branch must compress them to a single
    group button that sits comfortably on page 1 alongside the add/confirm
    actions."""

    damages = _build_identical_damages(damage_count)

    keyboards = damage_edit_keyboards_list(request_id="r-1", damages=damages)

    assert len(keyboards) == 1, (
        f"{damage_count} identical damages must collapse to one group — "
        f"got {len(keyboards)} keyboards instead."
    )
    total_buttons, total_rows = _count_buttons_and_rows(keyboards[0])
    # One group button + Добавить + Готово == 3.
    assert total_buttons == 3
    assert total_rows <= VK_INLINE_MAX_ROWS


@pytest.mark.parametrize("damage_count", [3, 9, 15, 30])
def test_distinct_damages_paginate_within_vk_limits(damage_count: int) -> None:
    """Distinct damage_types cannot be collapsed, so the worst case is
    ``damage_count`` group buttons. Pagination must keep every page within
    both the button and row caps — otherwise VK returns 911 and the user
    sees nothing. This is the invariant the old ``damage_edit_keyboard``
    violated by hard-capping at 8 and silently dropping the tail."""

    damages = _build_distinct_damages(damage_count)

    keyboards = damage_edit_keyboards_list(request_id="r-1", damages=damages)

    assert len(keyboards) >= 1
    total_group_buttons = 0
    for keyboard_json in keyboards:
        total_buttons, total_rows = _count_buttons_and_rows(keyboard_json)
        assert total_buttons <= VK_INLINE_MAX_BUTTONS, (
            f"page emitted {total_buttons} buttons, exceeds VK cap."
        )
        assert total_rows <= VK_INLINE_MAX_ROWS
        payload = json.loads(keyboard_json)
        for row in payload["buttons"]:
            for btn in row:
                raw = btn["action"]["payload"]
                p = json.loads(raw) if isinstance(raw, str) else raw
                if p.get("cmd") == "grp" and p.get("a") == "open":
                    total_group_buttons += 1
                elif p.get("cmd") == "edit" and p.get("a") == "edit_type":
                    total_group_buttons += 1
    # No silent drops: every distinct damage shows up as exactly one
    # actionable button across the whole keyboard set.
    assert total_group_buttons == damage_count
