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
    damage_edit_keyboard,
    edit_damage_type_keyboard,
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
    def test_contains_all_fourteen_parts_across_keyboards(self):
        keybs = part_selection_keyboards_list("req-1")
        all_buttons: list[dict] = []
        for kb_json in keybs:
            kb = _parse_keyboard(kb_json)
            all_buttons.extend(btn for row in kb["buttons"] for btn in row)
        assert len(all_buttons) == 14

    def test_each_button_has_part_cmd_and_request_id(self):
        keybs = part_selection_keyboards_list("req-1")
        for kb_json in keybs:
            kb = _parse_keyboard(kb_json)
            for btn in (b for row in kb["buttons"] for b in row):
                p = _payload(btn)
                assert p["cmd"] == "part"
                assert p["rid"] == "req-1"
                assert "pt" in p

    def test_first_button_is_bumper_front(self):
        first_kb = _parse_keyboard(part_selection_keyboards_list("req-1")[0])
        first = first_kb["buttons"][0][0]
        assert _payload(first)["pt"] == "bumper_front"

    def test_each_keyboard_fits_vk_inline_limits(self):
        for kb_json in part_selection_keyboards_list("req-1"):
            kb = _parse_keyboard(kb_json)
            rows = kb["buttons"]
            assert len(rows) <= 6
            all_btns = [b for row in rows for b in row]
            assert len(all_btns) <= 10


class TestDamageTypeSelectionKeyboard:
    def test_contains_five_damage_types(self):
        kb = _parse_keyboard(damage_type_selection_keyboard("req-1", "hood"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        assert len(all_buttons) == 5

    def test_button_payloads_include_part_type_and_damage_type(self):
        kb = _parse_keyboard(damage_type_selection_keyboard("req-1", "hood"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        for btn in all_buttons:
            p = _payload(btn)
            assert p["cmd"] == "dmg"
            assert p["rid"] == "req-1"
            assert p["pt"] == "hood"
            assert "dt" in p


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


class TestDamageEditKeyboard:
    def test_shows_per_damage_edit_and_delete_buttons(self):
        damages = [
            {"id": "d1", "part_type": "hood", "damage_type": "scratch"},
            {"id": "d2", "part_type": "trunk", "damage_type": "dent"},
        ]
        kb = _parse_keyboard(damage_edit_keyboard("req-1", damages))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        edit_btns = [b for b in all_buttons if _payload(b).get("a") == "edit_type"]
        delete_btns = [b for b in all_buttons if _payload(b).get("a") == "delete"]
        assert len(edit_btns) == 2
        assert len(delete_btns) == 2

    def test_includes_add_more_and_confirm_buttons(self):
        kb = _parse_keyboard(damage_edit_keyboard("req-1", []))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        cmds = {_payload(b)["cmd"] for b in all_buttons}
        assert "addmore" in cmds
        assert "confirm" in cmds


class TestEditDamageTypeKeyboard:
    def test_contains_all_five_damage_types(self):
        kb = _parse_keyboard(edit_damage_type_keyboard("req-1", "d1"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        assert len(all_buttons) == 5
        for btn in all_buttons:
            p = _payload(btn)
            assert p["cmd"] == "edtype"
            assert p["rid"] == "req-1"
            assert p["did"] == "d1"


class TestAddMoreOrConfirmKeyboard:
    def test_has_two_buttons_addmore_and_confirm(self):
        kb = _parse_keyboard(add_more_or_confirm_keyboard("req-1"))
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        assert len(all_buttons) == 2
        cmds = {_payload(b)["cmd"] for b in all_buttons}
        assert cmds == {"addmore", "confirm"}
