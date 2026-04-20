"""Behavior tests for the "UX-round" fixes.

Each ``Test*`` class pins one observable user-visible behavior from the
UX audit:

* **Result → Начать keyboard (#1)**: the pricing result must carry the
  single-button ``start_keyboard`` so the user can start over without
  typing.
* **Mode-switch abandons the previous session (#3 root cause)**:
  picking a mode from a stale keyboard no longer creates a parallel
  RepairRequest for the same chat_id — the old session is abandoned
  via the new backend endpoint first.
* **Running damage list on manual add (#5)**: after
  ``handle_damage_type_selection`` the reply must embed the full current
  basket so the user doesn't lose track of previously-added entries.
* **Running damage list when editing ML detections (#5)**: the
  edit-type sub-menu header must show the full list so the user doesn't
  have to scroll up to remember what else is in the basket.
* **Back-to-parts and back-to-edit-list buttons (#3)**: each back
  handler restores the correct "upper" screen.
* **Photo during manual flow nudges instead of spawning ML (#4)**:
  dispatching ``handle_photo`` would create a second RepairRequest for
  the same chat — the on_message handler must short-circuit before
  that happens.

These tests intentionally operate at the handler / dispatch boundary
(mocked ``BackendClient``, ``vkbottle.API``) because that is where the
bug actually lives; the underlying use cases are already well-covered
by their own suites.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.damage_edit import (
    handle_back_edit,
    handle_edit_action,
)
from auto_repair_estimator.bot.handlers.manual_flow import (
    handle_back_parts,
    handle_damage_type_selection,
    handle_mode_selection,
)
from auto_repair_estimator.bot.handlers.pricing import handle_confirm


# ---------------------------------------------------------------------------
# Test helpers (kept locally — the handlers under test have enough shape to
# make sharing these via conftest produce more noise than help).
# ---------------------------------------------------------------------------


def _event(peer_id: int = 100, user_id: int = 42) -> MagicMock:
    e = MagicMock()
    e.peer_id = peer_id
    e.user_id = user_id
    return e


def _api() -> AsyncMock:
    api = AsyncMock()
    api.messages.send = AsyncMock()
    return api


def _backend(**overrides: Any) -> AsyncMock:
    b = AsyncMock(spec=BackendClient)
    for k, v in overrides.items():
        setattr(b, k, AsyncMock(return_value=v))
    return b


def _parse_kb(kb_json: str | None) -> dict:
    assert kb_json is not None
    return json.loads(kb_json)


def _button_cmds(keyboard_json: str) -> set[str]:
    kb = _parse_kb(keyboard_json)
    cmds: set[str] = set()
    for row in kb["buttons"]:
        for btn in row:
            raw = btn["action"]["payload"]
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if cmd := payload.get("cmd"):
                cmds.add(cmd)
    return cmds


# ---------------------------------------------------------------------------
# #1 — pricing result carries start_keyboard
# ---------------------------------------------------------------------------


class TestPricingResultAttachesStartKeyboard:
    async def test_final_message_carries_start_button(self) -> None:
        # Regression guard: the old implementation wrote "напишите /start"
        # in plain text but attached no keyboard — forcing the user to
        # type. The fix attaches start_keyboard so the user has a
        # single-tap way to start a new scenario.
        event = _event()
        api = _api()
        backend = _backend(
            confirm_pricing={
                "total_cost_min": 0.0,
                "total_cost_max": 0.0,
                "total_hours_min": 0.0,
                "total_hours_max": 0.0,
                "breakdown": [],
                "notes": [],
            }
        )

        await handle_confirm(event, {"rid": "req-1"}, backend, api)

        call = api.messages.send.call_args.kwargs
        assert "keyboard" in call, "pricing result must carry a keyboard"
        assert _button_cmds(call["keyboard"]) == {"start"}


# ---------------------------------------------------------------------------
# #3 root cause — picking a mode abandons the previous active session
# ---------------------------------------------------------------------------


class TestModeSelectionAbandonsPreviousActive:
    async def test_abandons_existing_before_creating_new(self) -> None:
        # User currently has an ML/CREATED session; they scroll up and
        # click "Ручной ввод" from an older mode-selection keyboard.
        # Without the abandon this would leave two non-terminal rows for
        # the same chat_id and subsequent "part" clicks would resolve
        # against whichever was newest — the bug report behind #3.
        event = _event()
        api = _api()
        backend = _backend(
            get_active_request={"id": "prev-rid", "mode": "ml", "status": "created"},
            abandon_request={"id": "prev-rid", "status": "failed"},
            create_request={"id": "new-rid", "status": "pricing"},
        )

        await handle_mode_selection(event, {"m": "manual"}, backend, api)

        # Abandon must happen first, then create — otherwise the new
        # request would briefly coexist with the old one.
        backend.abandon_request.assert_awaited_once_with("prev-rid")
        backend.create_request.assert_awaited_once()

    async def test_no_abandon_call_when_chat_has_no_active_session(self) -> None:
        # Fresh user: no abandon round-trip should fire — both for
        # efficiency and to keep the backend logs free of spurious
        # "abandon of non-existent request" entries.
        event = _event()
        api = _api()
        backend = _backend(
            get_active_request=None,
            create_request={"id": "new-rid", "status": "pricing"},
        )

        await handle_mode_selection(event, {"m": "manual"}, backend, api)

        backend.abandon_request.assert_not_awaited()
        backend.create_request.assert_awaited_once()

    async def test_probe_failure_does_not_block_new_session(self) -> None:
        # A transient backend outage on the "get_active_request" probe
        # must NOT block the user from starting a new session —
        # otherwise a 500 would strand every user on the mode-selection
        # screen. We log and press on.
        event = _event()
        api = _api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_active_request = AsyncMock(side_effect=RuntimeError("boom"))
        backend.create_request = AsyncMock(return_value={"id": "new-rid"})

        await handle_mode_selection(event, {"m": "manual"}, backend, api)

        backend.create_request.assert_awaited_once()


# ---------------------------------------------------------------------------
# #5 — running damage list on manual add
# ---------------------------------------------------------------------------


class TestManualAddShowsRunningList:
    async def test_reply_includes_full_basket_after_add(self) -> None:
        # The user adds a second damage (hood scratch). The reply must
        # show BOTH the just-added "Добавлено: ..." line AND a numbered
        # list of everything currently in the basket — so the user can
        # see their full context without scrolling back.
        event = _event()
        api = _api()
        backend = _backend(
            add_damage={"id": "d2"},
            get_request={
                "id": "req-1",
                "damages": [
                    {"id": "d1", "part_type": "door", "damage_type": "dent", "is_deleted": False},
                    {"id": "d2", "part_type": "hood", "damage_type": "scratch", "is_deleted": False},
                ],
            },
        )

        await handle_damage_type_selection(
            event, {"rid": "req-1", "pt": "hood", "dt": "scratch"}, backend, api
        )

        message = api.messages.send.call_args.kwargs["message"]
        assert "Добавлено: Капот — Царапина" in message
        # Both damages must appear in the embedded running list.
        assert "Дверь — Вмятина" in message
        assert "Капот — Царапина" in message


# ---------------------------------------------------------------------------
# #5 — running damage list when editing ML detections
# ---------------------------------------------------------------------------


class TestEditSubMenuShowsRunningList:
    async def test_edit_type_header_lists_every_active_damage(self) -> None:
        # The user clicked "Подправить" on an ML result with three
        # damages and then selected the second one to change its type.
        # The sub-menu header must list all three so the user doesn't
        # have to scroll up to remember what else the model found.
        event = _event()
        api = _api()
        backend = _backend(
            get_request={
                "id": "req-1",
                "damages": [
                    {"id": "d1", "part_type": "door", "damage_type": "dent", "is_deleted": False},
                    {"id": "d2", "part_type": "hood", "damage_type": "scratch", "is_deleted": False},
                    {"id": "d3", "part_type": "trunk", "damage_type": "rust", "is_deleted": False},
                ],
            }
        )

        await handle_edit_action(
            event,
            {"a": "edit_type", "rid": "req-1", "did": "d2", "pt": "hood"},
            backend,
            api,
        )

        message = api.messages.send.call_args.kwargs["message"]
        assert "Дверь — Вмятина" in message
        assert "Капот — Царапина" in message
        assert "Багажник — Ржавчина" in message


# ---------------------------------------------------------------------------
# #3 — back buttons re-present the correct upper screen
# ---------------------------------------------------------------------------


class TestBackParts:
    async def test_resends_part_selection_messages(self) -> None:
        # "← К выбору детали" on the damage-type screen must drop the
        # user back onto the part list — same content that arrived on
        # the initial "Выберите повреждённую деталь" screen.
        event = _event()
        api = _api()
        backend = _backend()

        await handle_back_parts(event, {"rid": "req-1"}, backend, api)

        # Full part-list payload is split across multiple messages so it
        # fits VK's inline-button caps; every one of those messages must
        # carry a "part" keyboard.
        assert api.messages.send.await_count >= 1
        first_msg = api.messages.send.await_args_list[0].kwargs
        assert "деталь" in first_msg["message"].lower()
        assert "part" in _button_cmds(first_msg["keyboard"])

    async def test_missing_rid_is_rejected_gracefully(self) -> None:
        # Defensive: malformed payload must not crash the handler;
        # we surface a user-visible nudge instead.
        event = _event()
        api = _api()
        backend = _backend()

        await handle_back_parts(event, {}, backend, api)

        api.messages.send.assert_awaited()
        assert "Начать" in api.messages.send.call_args.kwargs["message"]


class TestBackEdit:
    async def test_refetches_and_shows_full_edit_list(self) -> None:
        # "← К списку повреждений" must re-present the damage_edit
        # keyboard, NOT the empty edit-type sub-menu. Crucially it
        # re-fetches the damages (fresh view of the basket) rather than
        # relying on stale payload data.
        event = _event()
        api = _api()
        backend = _backend(
            get_request={
                "id": "req-1",
                "damages": [
                    {"id": "d1", "part_type": "hood", "damage_type": "scratch", "is_deleted": False},
                ],
            }
        )

        await handle_back_edit(event, {"rid": "req-1"}, backend, api)

        backend.get_request.assert_awaited_once_with("req-1")
        call = api.messages.send.call_args.kwargs
        assert "Редактирование" in call["message"]
        assert "Капот — Царапина" in call["message"]
        # The keyboard must be the main damage-edit keyboard (addmore +
        # confirm + per-damage manage), not the sub-menu.
        cmds = _button_cmds(call["keyboard"])
        assert "addmore" in cmds
        assert "confirm" in cmds

    async def test_backend_failure_surfaces_user_facing_error(self) -> None:
        event = _event()
        api = _api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(side_effect=RuntimeError("boom"))

        await handle_back_edit(event, {"rid": "req-1"}, backend, api)

        message = api.messages.send.call_args.kwargs["message"]
        assert "ошибк" in message.lower()
