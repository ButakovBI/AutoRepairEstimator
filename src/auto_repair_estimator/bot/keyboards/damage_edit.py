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
from auto_repair_estimator.bot.damage_grouping import DamageGroup, group_damages
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS
from auto_repair_estimator.bot.vk_limits import (
    VK_INLINE_DEFAULT_BUTTONS_PER_ROW,
    VK_INLINE_MAX_BUTTONS,
)


def inference_result_keyboard(request_id: str) -> str:
    kb = Keyboard(inline=True)
    kb.add(Callback("Подтвердить", payload={"cmd": "confirm", "rid": request_id}), color=KeyboardButtonColor.POSITIVE)
    kb.add(Callback("Подправить", payload={"cmd": "edit", "a": "start_edit", "rid": request_id, "did": ""}))
    return kb.get_json()


# VK caps each inline keyboard at 10 buttons and 6 rows. The edit screen
# now paginates across multiple messages (damage_edit_keyboards_list) so
# we never silently drop groups past a hard cap. Per-page limits:
#
#   * non-last page: up to 10 group buttons (fills the keyboard).
#   * last page:     up to 8 group buttons + two trailing action buttons
#                    ("Добавить повреждение" and "Готово").
_GROUPS_PER_LAST_PAGE = VK_INLINE_MAX_BUTTONS - 2  # 8
_GROUPS_PER_INNER_PAGE = VK_INLINE_MAX_BUTTONS  # 10


def _group_button_label(group: DamageGroup) -> str:
    part_label = PART_LABELS.get(group.part_type, group.part_type or "?")
    damage_label = DAMAGE_LABELS.get(group.damage_type, group.damage_type or "?")
    if group.count > 1:
        # The "×N" suffix is the single signal the user gets that tapping
        # this button opens a group sub-menu rather than the per-damage
        # edit flow (bug: 17 scratches on a door overflowed VK limits).
        return f"{part_label} — {damage_label} (×{group.count})"
    return f"{part_label} — {damage_label}"


def _group_button_payload(request_id: str, group: DamageGroup) -> dict[str, Any]:
    """Route N=1 to the legacy single-damage sub-menu and N>1 to ``grp`` cmd.

    Keeping the N=1 path on the existing ``edit/edit_type`` cmd avoids a
    needless extra round-trip for the common "revise one damage" case
    and means existing tests and handlers keep working unchanged.
    """

    if group.count == 1:
        return {
            "cmd": "edit",
            "a": "edit_type",
            "rid": request_id,
            "did": group.damage_ids[0],
            "pt": group.part_type,
        }
    return {
        "cmd": "grp",
        "a": "open",
        "rid": request_id,
        "pt": group.part_type,
        "dt": group.damage_type,
    }


def _append_group_buttons(kb: Keyboard, request_id: str, groups: list[DamageGroup]) -> None:
    for i, group in enumerate(groups):
        kb.add(
            Callback(
                _group_button_label(group),
                payload=_group_button_payload(request_id, group),
            )
        )
        if (i + 1) % VK_INLINE_DEFAULT_BUTTONS_PER_ROW == 0 and i < len(groups) - 1:
            kb.row()


def _paginate_groups(groups: list[DamageGroup]) -> list[list[DamageGroup]]:
    """Split groups so each VK message stays within the inline-keyboard cap.

    All non-last pages are filled to ``_GROUPS_PER_INNER_PAGE`` buttons,
    leaving the trailing action buttons for the last page only. For the
    degenerate empty case we return a single empty page so the caller
    can still render the add/confirm controls.
    """

    if not groups:
        return [[]]

    pages: list[list[DamageGroup]] = []
    remaining = list(groups)
    while len(remaining) > _GROUPS_PER_LAST_PAGE:
        pages.append(remaining[:_GROUPS_PER_INNER_PAGE])
        remaining = remaining[_GROUPS_PER_INNER_PAGE:]
    pages.append(remaining)
    return pages


def damage_edit_keyboards_list(request_id: str, damages: list[dict[str, Any]]) -> list[str]:
    """Return one JSON keyboard per VK message for the edit flow.

    Replaces the old ``damage_edit_keyboard`` (single keyboard with a
    hard 8-damage cap). The last page is always the one with the
    "Добавить повреждение" / "Готово" buttons even when the list is
    empty, so the caller can emit it unconditionally.
    """

    groups = group_damages(damages)
    pages = _paginate_groups(groups)
    keyboards: list[str] = []
    for page_index, page_groups in enumerate(pages):
        is_last_page = page_index == len(pages) - 1
        kb = Keyboard(inline=True)
        if page_groups:
            _append_group_buttons(kb, request_id, page_groups)
        if is_last_page:
            if page_groups:
                kb.row()
            kb.add(Callback("Добавить повреждение", payload={"cmd": "addmore", "rid": request_id}))
            kb.add(
                Callback("Готово", payload={"cmd": "confirm", "rid": request_id}),
                color=KeyboardButtonColor.POSITIVE,
            )
        keyboards.append(kb.get_json())
    return keyboards


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


def group_submenu_keyboard(request_id: str, part_type: str, damage_type: str, count: int) -> str:
    """Sub-menu shown after tapping a grouped damage (N≥2) in the edit list.

    Offers bulk retype / bulk delete / single delete, plus a back button.
    The ``count`` is embedded into the button labels so the user sees the
    blast radius of each action before committing (VK doesn't render
    confirmation dialogs, so the label IS the confirmation).
    """

    kb = Keyboard(inline=True)
    kb.add(
        Callback(
            f"Изменить тип всем (×{count})",
            payload={"cmd": "grp", "a": "retype", "rid": request_id, "pt": part_type, "dt": damage_type},
        )
    )
    kb.row()
    kb.add(
        Callback(
            f"Удалить все (×{count})",
            payload={"cmd": "grp", "a": "del_all", "rid": request_id, "pt": part_type, "dt": damage_type},
        ),
        color=KeyboardButtonColor.NEGATIVE,
    )
    kb.row()
    kb.add(
        Callback(
            "Удалить одно",
            payload={"cmd": "grp", "a": "del_one", "rid": request_id, "pt": part_type, "dt": damage_type},
        )
    )
    kb.row()
    kb.add(
        Callback("← К списку повреждений", payload={"cmd": "back_edit", "rid": request_id}),
        color=KeyboardButtonColor.SECONDARY,
    )
    return kb.get_json()


def group_retype_keyboard(request_id: str, part_type: str, damage_type_old: str) -> str:
    """Type-picker for "Изменить тип всем": applies to every damage in group.

    Filters by the part's compatibility set so we never offer "flat_tire"
    on a door. The currently-set damage type is hidden — picking it would
    be a no-op and just clutters the list.
    """

    kb = Keyboard(inline=True)
    allowed_values = [v for v in _allowed_damage_values_for(part_type) if v != damage_type_old]
    # Defensive fallback: if the filter produced an empty list (e.g. the
    # part has only one compatible damage type and that's the current one)
    # keep the old value in so the user at least has a visible escape
    # hatch; tapping it is a no-op but it prevents rendering an empty kb.
    if not allowed_values:
        allowed_values = _allowed_damage_values_for(part_type)
    for i, damage_type in enumerate(allowed_values):
        label = DAMAGE_LABELS.get(damage_type, damage_type)
        kb.add(
            Callback(
                label,
                payload={
                    "cmd": "grp",
                    "a": "apply_retype",
                    "rid": request_id,
                    "pt": part_type,
                    "dt": damage_type_old,
                    "nd": damage_type,
                },
            )
        )
        if i % 2 == 1 and i < len(allowed_values) - 1:
            kb.row()
    kb.row()
    kb.add(
        Callback("← К списку повреждений", payload={"cmd": "back_edit", "rid": request_id}),
        color=KeyboardButtonColor.SECONDARY,
    )
    return kb.get_json()


def add_more_or_confirm_keyboard(request_id: str) -> str:
    """Keyboard shown after adding a damage in the manual flow.

    ``Подправить`` is wired to the same ``edit/start_edit`` cmd as the ML
    flow uses, so manual and ML share one edit UX. Before this change,
    a user who misclicked in manual mode had no way to revise their
    choice without confirming-then-starting-over.
    """

    kb = Keyboard(inline=True)
    kb.add(Callback("Добавить ещё", payload={"cmd": "addmore", "rid": request_id}))
    kb.add(
        Callback("Подправить", payload={"cmd": "edit", "a": "start_edit", "rid": request_id, "did": ""}),
    )
    kb.row()
    kb.add(
        Callback("Подтвердить", payload={"cmd": "confirm", "rid": request_id}),
        color=KeyboardButtonColor.POSITIVE,
    )
    return kb.get_json()
