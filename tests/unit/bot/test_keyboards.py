"""Behavioral tests for VK bot keyboards.

Each test verifies a single observable behavior of the keyboard factory functions:
- Correct button labels
- Correct payload structure for callback dispatch
- Correct layout (inline keyboard)
"""

from __future__ import annotations

import json

from auto_repair_estimator.bot.keyboards.damage_edit import (
    add_more_or_confirm_keyboard,
    damage_edit_keyboards_list,
    edit_damage_type_keyboard,
    group_retype_keyboard,
    group_submenu_keyboard,
    inference_result_keyboard,
)
from auto_repair_estimator.bot.keyboards.damage_type_selection import damage_type_selection_keyboard
from auto_repair_estimator.bot.keyboards.mode_selection import mode_selection_keyboard
from auto_repair_estimator.bot.keyboards.part_selection import part_selection_keyboards_list


def _parse_keyboard(kb_json: str) -> dict:
    return json.loads(kb_json)


def _payload(btn: dict) -> dict:
    raw = btn["action"]["payload"]
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


class TestModeSelectionKeyboard:
    def test_returns_two_buttons(self):
        kb = _parse_keyboard(mode_selection_keyboard())
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        assert len(all_buttons) == 2

    def test_ml_button_payload_has_mode_ml(self):
        kb = _parse_keyboard(mode_selection_keyboard())
        first_button = kb["buttons"][0][0]
        assert _payload(first_button) == {"cmd": "mode", "m": "ml"}

    def test_manual_button_payload_has_mode_manual(self):
        kb = _parse_keyboard(mode_selection_keyboard())
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        manual_btn = all_buttons[1]
        assert _payload(manual_btn) == {"cmd": "mode", "m": "manual"}

    def test_keyboard_is_inline(self):
        kb = _parse_keyboard(mode_selection_keyboard())
        assert kb["inline"] is True


class TestPartSelectionKeyboard:
    def test_contains_all_twelve_parts_across_keyboards(self):
        keybs = part_selection_keyboards_list("req-1")
        all_buttons: list[dict] = []
        for kb_json in keybs:
            kb = _parse_keyboard(kb_json)
            all_buttons.extend(btn for row in kb["buttons"] for btn in row)
        assert len(all_buttons) == 12

    def test_each_button_has_part_cmd_and_request_id(self):
        keybs = part_selection_keyboards_list("req-1")
        for kb_json in keybs:
            kb = _parse_keyboard(kb_json)
            for btn in (b for row in kb["buttons"] for b in row):
                p = _payload(btn)
                assert p["cmd"] == "part"
                assert p["rid"] == "req-1"
                assert "pt" in p

    def test_first_button_is_door(self):
        first_kb = _parse_keyboard(part_selection_keyboards_list("req-1")[0])
        first = first_kb["buttons"][0][0]
        assert _payload(first)["pt"] == "door"

    def test_each_keyboard_fits_vk_inline_limits(self):
        from auto_repair_estimator.bot.vk_limits import VK_INLINE_MAX_BUTTONS, VK_INLINE_MAX_ROWS

        for kb_json in part_selection_keyboards_list("req-1"):
            kb = _parse_keyboard(kb_json)
            rows = kb["buttons"]
            assert len(rows) <= VK_INLINE_MAX_ROWS
            all_btns = [b for row in rows for b in row]
            assert len(all_btns) <= VK_INLINE_MAX_BUTTONS


def _damage_buttons(keyboard_json: str) -> list[dict]:
    """Return only the damage-selection buttons (``cmd=dmg``).

    The keyboard also carries a "← К выбору детали" back button that is
    irrelevant to damage-compatibility assertions; filtering by cmd
    keeps those tests focused on what they're actually verifying.
    """

    kb = _parse_keyboard(keyboard_json)
    return [
        btn
        for row in kb["buttons"]
        for btn in row
        if _payload(btn).get("cmd") == "dmg"
    ]


class TestDamageTypeSelectionKeyboard:
    def test_body_part_shows_five_compatible_damage_types(self):
        # Body panels (hood etc.) accept scratch, dent, paint_chip, rust, crack
        # per PART_DAMAGE_COMPATIBILITY -- glass/wheel/headlight damage types
        # must NOT be offered here because they can't be priced on a body panel.
        dmg_btns = _damage_buttons(damage_type_selection_keyboard("req-1", "hood"))
        dt_values = {_payload(b)["dt"] for b in dmg_btns}
        assert dt_values == {"scratch", "dent", "paint_chip", "rust", "crack"}

    def test_headlight_shows_only_broken_headlight(self):
        # Headlight has exactly one priceable damage type: broken_headlight.
        # Previously the UI exposed scratch/crack/etc. on headlights, creating
        # entries that PricingService could only surface as "not priceable" notes.
        dmg_btns = _damage_buttons(damage_type_selection_keyboard("req-1", "headlight"))
        assert len(dmg_btns) == 1
        assert _payload(dmg_btns[0])["dt"] == "broken_headlight"

    def test_wheel_shows_only_flat_tire(self):
        # Wheel damage routes to a tyre shop via TYRE_SHOP_NOTE; "flat_tire"
        # is the single damage type the UI should surface for a wheel.
        dmg_btns = _damage_buttons(damage_type_selection_keyboard("req-1", "wheel"))
        assert len(dmg_btns) == 1
        assert _payload(dmg_btns[0])["dt"] == "flat_tire"

    def test_glass_parts_show_only_broken_glass(self):
        # All three glass surfaces share identical compatibility.
        for glass_part in ("front_windshield", "rear_windshield", "side_window"):
            dmg_btns = _damage_buttons(damage_type_selection_keyboard("req-1", glass_part))
            assert len(dmg_btns) == 1, f"{glass_part} should have exactly 1 damage type"
            assert _payload(dmg_btns[0])["dt"] == "broken_glass"

    def test_button_payloads_include_part_type_and_damage_type(self):
        dmg_btns = _damage_buttons(damage_type_selection_keyboard("req-1", "hood"))
        for btn in dmg_btns:
            p = _payload(btn)
            assert p["cmd"] == "dmg"
            assert p["rid"] == "req-1"
            assert p["pt"] == "hood"
            assert "dt" in p

    def test_unknown_part_renders_no_damage_buttons(self):
        # Unknown part_type (legacy/injected callback) must not crash the
        # handler; we render zero damage buttons and let the backend reject
        # whatever the client might post anyway. The Back affordance is
        # still present so the user has a deterministic escape hatch.
        kb_json = damage_type_selection_keyboard("req-1", "not_a_part")
        assert _damage_buttons(kb_json) == []

    def test_back_to_parts_button_is_always_present(self):
        # Bug #3 (UX round): the damage-type screen is a dead end without a
        # "back" button — the user has already committed to a part and there
        # is no natural way to reconsider without scrolling up. The back
        # button must be rendered for every part, including the 1-option
        # screens (headlight/wheel/glass), so the escape hatch is uniform.
        for part in ("hood", "headlight", "wheel", "front_windshield", "not_a_part"):
            kb = _parse_keyboard(damage_type_selection_keyboard("req-1", part))
            all_buttons = [b for row in kb["buttons"] for b in row]
            back = [b for b in all_buttons if _payload(b).get("cmd") == "back_parts"]
            assert len(back) == 1, f"{part}: expected one back_parts button"
            assert _payload(back[0])["rid"] == "req-1"


class TestInferenceResultKeyboard:
    def test_has_confirm_and_edit_buttons(self):
        kb = _parse_keyboard(inference_result_keyboard("req-1"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        assert len(all_buttons) == 2
        cmds = {_payload(b)["cmd"] for b in all_buttons}
        assert cmds == {"confirm", "edit"}

    def test_confirm_payload_has_request_id(self):
        kb = _parse_keyboard(inference_result_keyboard("req-1"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        confirm_btn = [b for b in all_buttons if _payload(b)["cmd"] == "confirm"][0]
        assert _payload(confirm_btn)["rid"] == "req-1"


def _all_buttons_across_keyboards(keyboards: list[str]) -> list[dict]:
    out: list[dict] = []
    for kb_json in keyboards:
        kb = _parse_keyboard(kb_json)
        out.extend(btn for row in kb["buttons"] for btn in row)
    return out


class TestDamageEditKeyboardsList:
    def test_single_damage_emits_legacy_edit_type_button(self):
        # N=1 must stay on the legacy per-damage edit path so the user
        # keeps the unchanged "edit / delete / back" sub-menu experience.
        damages = [{"id": "d1", "part_type": "hood", "damage_type": "scratch"}]
        kbs = damage_edit_keyboards_list("req-1", damages)
        assert len(kbs) == 1
        buttons = _all_buttons_across_keyboards(kbs)
        edit_btns = [b for b in buttons if _payload(b).get("a") == "edit_type"]
        assert len(edit_btns) == 1
        assert _payload(edit_btns[0])["did"] == "d1"

    def test_grouping_collapses_duplicate_part_damage_pairs(self):
        # The reported bug: 17 scratches on a door collapsed to ONE
        # button with "×17" suffix — otherwise the VK 10-button cap was
        # silently eating damages past position #8.
        damages = [
            {"id": f"d{i}", "part_type": "door", "damage_type": "scratch"}
            for i in range(17)
        ]
        kbs = damage_edit_keyboards_list("req-1", damages)
        buttons = _all_buttons_across_keyboards(kbs)
        group_btns = [b for b in buttons if _payload(b).get("cmd") == "grp"]
        assert len(group_btns) == 1, "17 identical damages must collapse to one group"
        label = group_btns[0]["action"]["label"]
        assert "×17" in label
        assert "Дверь" in label
        assert "Царапина" in label

    def test_group_button_payload_targets_grp_open(self):
        damages = [
            {"id": "d1", "part_type": "hood", "damage_type": "scratch"},
            {"id": "d2", "part_type": "hood", "damage_type": "scratch"},
        ]
        kbs = damage_edit_keyboards_list("req-1", damages)
        group_btns = [
            b for b in _all_buttons_across_keyboards(kbs) if _payload(b).get("cmd") == "grp"
        ]
        assert len(group_btns) == 1
        payload = _payload(group_btns[0])
        assert payload == {
            "cmd": "grp",
            "a": "open",
            "rid": "req-1",
            "pt": "hood",
            "dt": "scratch",
        }

    def test_empty_damages_still_shows_add_and_confirm(self):
        # Edge case: editing an empty basket — the screen must still
        # render the action buttons so the user has a clear exit.
        kbs = damage_edit_keyboards_list("req-1", [])
        assert len(kbs) == 1
        buttons = _all_buttons_across_keyboards(kbs)
        cmds = {_payload(b)["cmd"] for b in buttons}
        assert cmds == {"addmore", "confirm"}

    def test_every_page_respects_vk_inline_limits(self):
        # 25 distinct groups — worst-case spread we can build from the
        # domain (12 parts × all their compatible damages). Each page
        # must stay within VK's 10-button cap and 6-row cap.
        from auto_repair_estimator.bot.vk_limits import (
            VK_INLINE_MAX_BUTTONS,
            VK_INLINE_MAX_ROWS,
        )

        damages = [
            {"id": f"d{i}", "part_type": "door", "damage_type": f"fake_{i}"}
            for i in range(25)
        ]
        kbs = damage_edit_keyboards_list("req-1", damages)
        # Every damage is distinct → must produce ≥ 3 pages (25 / 8 LAST=3 or 2+1 overflow).
        assert len(kbs) >= 3
        for kb_json in kbs:
            kb = _parse_keyboard(kb_json)
            rows = kb["buttons"]
            assert len(rows) <= VK_INLINE_MAX_ROWS
            btns = [b for row in rows for b in row]
            assert len(btns) <= VK_INLINE_MAX_BUTTONS

    def test_only_last_page_carries_addmore_and_confirm(self):
        damages = [
            {"id": f"d{i}", "part_type": "door", "damage_type": f"fake_{i}"}
            for i in range(25)
        ]
        kbs = damage_edit_keyboards_list("req-1", damages)
        for kb_json in kbs[:-1]:
            buttons = _all_buttons_across_keyboards([kb_json])
            cmds = {_payload(b)["cmd"] for b in buttons}
            assert "addmore" not in cmds
            assert "confirm" not in cmds
        # Last page carries both.
        last_cmds = {
            _payload(b)["cmd"] for b in _all_buttons_across_keyboards([kbs[-1]])
        }
        assert "addmore" in last_cmds
        assert "confirm" in last_cmds


class TestGroupSubmenuKeyboard:
    def test_has_three_action_buttons_plus_back(self):
        kb = _parse_keyboard(group_submenu_keyboard("req-1", "door", "scratch", 5))
        buttons = [b for row in kb["buttons"] for b in row]
        actions = {_payload(b).get("a") for b in buttons}
        assert {"retype", "del_all", "del_one"} <= actions
        cmds = {_payload(b)["cmd"] for b in buttons}
        assert "back_edit" in cmds

    def test_labels_include_group_count_so_user_sees_blast_radius(self):
        kb = _parse_keyboard(group_submenu_keyboard("req-1", "door", "scratch", 7))
        labels = [b["action"]["label"] for row in kb["buttons"] for b in row]
        # "×7" must appear on the destructive buttons — without it the
        # user cannot tell they're about to nuke seven items, not one.
        assert any("×7" in lbl for lbl in labels)


class TestGroupRetypeKeyboard:
    def test_filters_by_part_and_hides_current_type(self):
        # Door is a body panel: compatible with scratch/dent/paint_chip/rust/crack.
        # The old type ``scratch`` must be filtered out so the user cannot
        # apply a no-op retype (which would waste a VK button slot).
        kb = _parse_keyboard(group_retype_keyboard("req-1", "door", "scratch"))
        buttons = [b for row in kb["buttons"] for b in row]
        retype_btns = [b for b in buttons if _payload(b).get("a") == "apply_retype"]
        dt_old_values = {_payload(b)["dt"] for b in retype_btns}
        assert dt_old_values == {"scratch"}, "dt carries the OLD type for resolution"
        nd_values = {_payload(b)["nd"] for b in retype_btns}
        assert "scratch" not in nd_values
        assert nd_values == {"dent", "paint_chip", "rust", "crack"}

    def test_retype_button_payload_shape(self):
        kb = _parse_keyboard(group_retype_keyboard("req-1", "hood", "dent"))
        btn = [
            b for row in kb["buttons"] for b in row if _payload(b).get("a") == "apply_retype"
        ][0]
        payload = _payload(btn)
        assert payload["cmd"] == "grp"
        assert payload["rid"] == "req-1"
        assert payload["pt"] == "hood"
        assert payload["dt"] == "dent"  # OLD
        assert "nd" in payload  # NEW


class TestEditDamageTypeKeyboard:
    def test_legacy_call_without_part_shows_full_list_and_delete(self):
        # Backwards-compat path: if a caller omits part_type (legacy payload
        # that predates the pt-in-payload change) we degrade to the full
        # 8-type list + delete rather than rendering an empty keyboard.
        kb = _parse_keyboard(edit_damage_type_keyboard("req-1", "d1"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        type_btns = [b for b in all_buttons if _payload(b)["cmd"] == "edtype"]
        delete_btns = [b for b in all_buttons if _payload(b).get("a") == "delete"]
        assert len(type_btns) == 8
        assert len(delete_btns) == 1

    def test_filters_by_part_for_headlight(self):
        # With a known part_type the sub-keyboard must mirror the forward
        # selection keyboard's filter -- otherwise the user can "edit" a
        # headlight damage into an incompatible type and get a 400 from the
        # backend validator.
        kb = _parse_keyboard(edit_damage_type_keyboard("req-1", "d1", "headlight"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        type_btns = [b for b in all_buttons if _payload(b)["cmd"] == "edtype"]
        assert len(type_btns) == 1
        assert _payload(type_btns[0])["dt"] == "broken_headlight"

    def test_filters_by_part_for_body_panel(self):
        kb = _parse_keyboard(edit_damage_type_keyboard("req-1", "d1", "door"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        type_btns = [b for b in all_buttons if _payload(b)["cmd"] == "edtype"]
        dt_values = {_payload(b)["dt"] for b in type_btns}
        assert dt_values == {"scratch", "dent", "paint_chip", "rust", "crack"}

    def test_propagates_request_and_damage_ids(self):
        kb = _parse_keyboard(edit_damage_type_keyboard("req-1", "d1", "hood"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        type_btns = [b for b in all_buttons if _payload(b)["cmd"] == "edtype"]
        for btn in type_btns:
            p = _payload(btn)
            assert p["rid"] == "req-1"
            assert p["did"] == "d1"


class TestDamageEditKeyboardPropagatesPartType:
    def test_single_damage_edit_callback_carries_part_type(self):
        # The N=1 path uses the legacy edit_type cmd and must carry `pt`
        # so the sub-handler can render the part-filtered edit keyboard
        # without a backend round-trip.
        damages = [{"id": "d1", "part_type": "wheel", "damage_type": "flat_tire"}]
        kbs = damage_edit_keyboards_list("req-1", damages)
        buttons = _all_buttons_across_keyboards(kbs)
        manage_btns = [b for b in buttons if _payload(b).get("a") == "edit_type"]
        assert len(manage_btns) == 1
        assert _payload(manage_btns[0])["pt"] == "wheel"


class TestAddMoreOrConfirmKeyboard:
    def test_has_three_buttons_addmore_edit_and_confirm(self):
        # "Подправить" is the manual-flow entry into the shared edit
        # screen — without it, manual-mode users cannot revise their
        # additions before confirming (they had to reset the session).
        kb = _parse_keyboard(add_more_or_confirm_keyboard("req-1"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        cmds = {_payload(b)["cmd"] for b in all_buttons}
        assert "addmore" in cmds
        assert "confirm" in cmds
        edit_btns = [b for b in all_buttons if _payload(b).get("cmd") == "edit"]
        assert len(edit_btns) == 1
        assert _payload(edit_btns[0])["a"] == "start_edit"
