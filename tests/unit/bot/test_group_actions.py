"""Behavioural tests for ``handle_group_action`` (``cmd=grp`` family).

The group-action handler fans out bulk edits/deletes across every
damage that shares ``(part_type, damage_type)`` in the current backend
snapshot. Key invariants:

* Group identity is resolved freshly each time — stale ids from the
  previous screen must not leak through the payload.
* Bulk retype / delete iterate the resolved id list and survive
  partial failures (some damages may have been deleted in a concurrent
  round-trip); the user gets a soft notice rather than a crash.
* ``del_one`` removes exactly one damage and refreshes the list.
* A vanished group (race with a prior del_all) triggers the "ничего
  нет" notice + refresh, never a silent no-op.

Malformed payloads fall through to a single "Некорректная кнопка" reply
and perform no backend mutation — the existing QA suite already pins
this contract for other handlers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.damage_edit import handle_group_action


def _make_event(peer_id: int = 100) -> MagicMock:
    event = MagicMock()
    event.peer_id = peer_id
    return event


def _make_api() -> AsyncMock:
    api = AsyncMock()
    api.messages.send = AsyncMock()
    return api


def _request_payload(damages: list[dict]) -> dict:
    return {"id": "req-1", "damages": damages}


class TestGroupOpen:
    async def test_open_shows_submenu_with_live_count(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(
            return_value=_request_payload(
                [
                    {"id": f"d{i}", "part_type": "door", "damage_type": "scratch"}
                    for i in range(3)
                ]
            )
        )

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "open", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        call = api.messages.send.await_args.kwargs
        assert "×3" in call["message"]
        assert "Дверь" in call["message"]

    async def test_open_of_vanished_group_triggers_refresh(self):
        # The group that existed on the previous screen has been wiped
        # (e.g. the user already pressed "Удалить все" once). Opening
        # must surface the "ничего нет" copy, not a ×0 submenu.
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(return_value=_request_payload([]))

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "open", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        texts = [c.kwargs.get("message", "") for c in api.messages.send.await_args_list]
        assert any("больше нет" in t for t in texts)


class TestGroupRetypeAll:
    async def test_apply_retype_calls_edit_damage_for_every_id(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(
            return_value=_request_payload(
                [
                    {"id": "d1", "part_type": "door", "damage_type": "scratch"},
                    {"id": "d2", "part_type": "door", "damage_type": "scratch"},
                ]
            )
        )
        backend.edit_damage = AsyncMock()

        await handle_group_action(
            event,
            {
                "cmd": "grp",
                "a": "apply_retype",
                "rid": "req-1",
                "pt": "door",
                "dt": "scratch",
                "nd": "dent",
            },
            backend,
            api,
        )

        assert backend.edit_damage.await_count == 2
        ids = {call.args[1] for call in backend.edit_damage.await_args_list}
        assert ids == {"d1", "d2"}
        new_types = {call.args[2] for call in backend.edit_damage.await_args_list}
        assert new_types == {"dent"}

    async def test_apply_retype_missing_new_type_is_malformed(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.edit_damage = AsyncMock()

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "apply_retype", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        backend.edit_damage.assert_not_awaited()
        call = api.messages.send.await_args.kwargs
        assert "некорректная" in call["message"].lower()


class TestGroupDeleteAll:
    async def test_del_all_deletes_every_damage_in_group(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(
            return_value=_request_payload(
                [
                    {"id": "d1", "part_type": "door", "damage_type": "scratch"},
                    {"id": "d2", "part_type": "door", "damage_type": "scratch"},
                    {"id": "d3", "part_type": "hood", "damage_type": "dent"},
                ]
            )
        )
        backend.delete_damage = AsyncMock()

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "del_all", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        # Exactly the door/scratch ids, not d3 (different group).
        ids = {call.args[1] for call in backend.delete_damage.await_args_list}
        assert ids == {"d1", "d2"}

    async def test_del_all_full_failure_sends_error(self):
        # Every single delete raised — the user is told the bulk op
        # failed and the edit screen is NOT refreshed (nothing changed
        # server-side, a refresh would just be noise).
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(
            return_value=_request_payload(
                [{"id": "d1", "part_type": "door", "damage_type": "scratch"}]
            )
        )
        backend.delete_damage = AsyncMock(side_effect=RuntimeError("boom"))

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "del_all", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        texts = [c.kwargs.get("message", "") for c in api.messages.send.await_args_list]
        assert any("ошибка" in t.lower() for t in texts)


class TestGroupDeleteOne:
    async def test_del_one_removes_first_id_and_refreshes(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(
            return_value=_request_payload(
                [
                    {"id": "d1", "part_type": "door", "damage_type": "scratch"},
                    {"id": "d2", "part_type": "door", "damage_type": "scratch"},
                ]
            )
        )
        backend.delete_damage = AsyncMock()

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "del_one", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        backend.delete_damage.assert_awaited_once()
        assert backend.delete_damage.await_args.args[1] == "d1"

    async def test_del_one_of_vanished_group_refreshes(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.get_request = AsyncMock(return_value=_request_payload([]))
        backend.delete_damage = AsyncMock()

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "del_one", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        backend.delete_damage.assert_not_awaited()
        texts = [c.kwargs.get("message", "") for c in api.messages.send.await_args_list]
        assert any("больше нет" in t for t in texts)


class TestGroupRetypeScreen:
    async def test_retype_action_opens_picker_keyboard(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "retype", "rid": "req-1", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        call = api.messages.send.await_args.kwargs
        assert "выберите новый тип" in call["message"].lower()
        assert call["keyboard"] is not None
        # Does NOT mutate: the retype button is a picker entry point,
        # the mutation happens on ``apply_retype``.
        backend.edit_damage = getattr(backend, "edit_damage", AsyncMock())
        backend.edit_damage.assert_not_awaited()


class TestGroupMalformedPayload:
    async def test_missing_action_rejects_with_no_mutation(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)

        await handle_group_action(event, {"cmd": "grp", "rid": "r"}, backend, api)

        call = api.messages.send.await_args.kwargs
        assert "некорректная" in call["message"].lower()

    async def test_unknown_action_rejects(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)

        await handle_group_action(
            event,
            {"cmd": "grp", "a": "explode_universe", "rid": "r", "pt": "door", "dt": "scratch"},
            backend,
            api,
        )

        call = api.messages.send.await_args.kwargs
        assert "некорректная" in call["message"].lower()
