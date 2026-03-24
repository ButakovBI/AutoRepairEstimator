"""Behavioral tests for VK bot handler functions.

Tests verify that each handler:
- Calls the correct backend methods with the right arguments
- Sends the appropriate VK messages via the API
- Handles error cases gracefully
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.damage_edit import (
    handle_edit_action,
    handle_edit_damage_type,
)
from auto_repair_estimator.bot.handlers.manual_flow import (
    handle_add_more,
    handle_damage_type_selection,
    handle_mode_selection,
    handle_part_selection,
)
from auto_repair_estimator.bot.handlers.pricing import handle_confirm
from auto_repair_estimator.bot.handlers.start import handle_start


def _make_event(peer_id: int = 100, user_id: int = 42) -> MagicMock:
    event = MagicMock()
    event.peer_id = peer_id
    event.user_id = user_id
    return event


def _make_api() -> AsyncMock:
    api = AsyncMock()
    api.messages.send = AsyncMock()
    return api


def _make_backend(**overrides) -> AsyncMock:
    backend = AsyncMock(spec=BackendClient)
    for k, v in overrides.items():
        setattr(backend, k, AsyncMock(return_value=v))
    return backend


class TestHandleStart:
    async def test_sends_welcome_message_with_mode_keyboard(self):
        message = MagicMock()
        message.answer = AsyncMock()

        await handle_start(message)

        message.answer.assert_called_once()
        call_kwargs = message.answer.call_args
        assert "keyboard" in call_kwargs.kwargs or (len(call_kwargs.args) > 1)
        text = call_kwargs.kwargs.get("message") or call_kwargs.args[0]
        assert "Добро пожаловать" in text


class TestHandleModeSelection:
    async def test_manual_mode_creates_request_and_sends_part_keyboard(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend(create_request={"id": "req-1", "status": "pricing"})

        await handle_mode_selection(event, {"m": "manual"}, backend, api)

        backend.create_request.assert_awaited_once_with(chat_id=100, user_id=42, mode="manual")
        assert api.messages.send.await_count == 2
        first = api.messages.send.await_args_list[0].kwargs
        second = api.messages.send.await_args_list[1].kwargs
        assert "Выберите повреждённую деталь" in first["message"]
        kb1 = json.loads(first["keyboard"])
        kb2 = json.loads(second["keyboard"])
        all_btns = [b for row in kb1["buttons"] for b in row] + [b for row in kb2["buttons"] for b in row]
        assert len(all_btns) == 14

    async def test_ml_mode_creates_request_and_asks_for_photo(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend(create_request={"id": "req-2", "status": "created"})

        await handle_mode_selection(event, {"m": "ml"}, backend, api)

        backend.create_request.assert_awaited_once_with(chat_id=100, user_id=42, mode="ml")
        call_kwargs = api.messages.send.call_args.kwargs
        assert "фотографию" in call_kwargs["message"]

    async def test_backend_error_sends_error_message(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.create_request = AsyncMock(side_effect=Exception("connection error"))

        await handle_mode_selection(event, {"m": "manual"}, backend, api)

        call_kwargs = api.messages.send.call_args.kwargs
        assert "ошибка" in call_kwargs["message"].lower()


class TestHandlePartSelection:
    async def test_sends_damage_type_keyboard_for_selected_part(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend()

        await handle_part_selection(event, {"rid": "req-1", "pt": "hood"}, backend, api)

        call_kwargs = api.messages.send.call_args.kwargs
        assert "Капот" in call_kwargs["message"]
        keyboard = json.loads(call_kwargs["keyboard"])
        all_btns = [b for row in keyboard["buttons"] for b in row]
        assert len(all_btns) == 5


class TestHandleDamageTypeSelection:
    async def test_adds_damage_and_sends_confirmation(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend(add_damage={"id": "d1"})

        await handle_damage_type_selection(event, {"rid": "req-1", "pt": "hood", "dt": "scratch"}, backend, api)

        backend.add_damage.assert_awaited_once_with(request_id="req-1", part_type="hood", damage_type="scratch")
        call_kwargs = api.messages.send.call_args.kwargs
        assert "Добавлено" in call_kwargs["message"]
        assert "Капот" in call_kwargs["message"]
        assert "Царапина" in call_kwargs["message"]

    async def test_backend_error_sends_error_message(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.add_damage = AsyncMock(side_effect=Exception("fail"))

        await handle_damage_type_selection(event, {"rid": "req-1", "pt": "hood", "dt": "scratch"}, backend, api)

        call_kwargs = api.messages.send.call_args.kwargs
        assert "ошибка" in call_kwargs["message"].lower()


class TestHandleAddMore:
    async def test_sends_part_selection_keyboard(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend()

        await handle_add_more(event, {"rid": "req-1"}, backend, api)

        assert api.messages.send.await_count == 2
        first = api.messages.send.await_args_list[0].kwargs
        assert "деталь" in first["message"].lower()
        kb1 = json.loads(first["keyboard"])
        kb2 = json.loads(api.messages.send.await_args_list[1].kwargs["keyboard"])
        all_btns = [b for row in kb1["buttons"] for b in row] + [b for row in kb2["buttons"] for b in row]
        assert len(all_btns) == 14


class TestHandleEditAction:
    async def test_start_edit_fetches_damages_and_shows_edit_keyboard(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend(
            get_request={
                "id": "req-1",
                "damages": [
                    {"id": "d1", "part_type": "hood", "damage_type": "scratch", "is_deleted": False},
                ],
            }
        )

        await handle_edit_action(event, {"a": "start_edit", "rid": "req-1", "did": ""}, backend, api)

        backend.get_request.assert_awaited_once_with("req-1")
        call_kwargs = api.messages.send.call_args.kwargs
        assert "Редактирование" in call_kwargs["message"]

    async def test_edit_type_sends_damage_type_keyboard(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend()

        await handle_edit_action(event, {"a": "edit_type", "rid": "req-1", "did": "d1"}, backend, api)

        call_kwargs = api.messages.send.call_args.kwargs
        assert "тип повреждения" in call_kwargs["message"].lower()

    async def test_delete_action_removes_damage_and_refreshes_list(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend(
            get_request={"id": "req-1", "damages": []},
        )
        backend.delete_damage = AsyncMock()

        await handle_edit_action(event, {"a": "delete", "rid": "req-1", "did": "d1"}, backend, api)

        backend.delete_damage.assert_awaited_once_with("req-1", "d1")
        call_kwargs = api.messages.send.call_args.kwargs
        assert "удалено" in call_kwargs["message"].lower()


class TestHandleEditDamageType:
    async def test_updates_damage_and_refreshes_list(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend(
            edit_damage={"id": "d1", "damage_type": "dent"},
            get_request={
                "id": "req-1",
                "damages": [{"id": "d1", "part_type": "hood", "damage_type": "dent", "is_deleted": False}],
            },
        )

        await handle_edit_damage_type(event, {"rid": "req-1", "did": "d1", "dt": "dent"}, backend, api)

        backend.edit_damage.assert_awaited_once_with("req-1", "d1", "dent")
        call_kwargs = api.messages.send.call_args.kwargs
        assert "обновлён" in call_kwargs["message"].lower()


class TestHandleConfirm:
    async def test_confirms_pricing_and_sends_result(self):
        event = _make_event()
        api = _make_api()
        backend = _make_backend(
            confirm_pricing={
                "total_cost": 5000.0,
                "total_hours": 3.5,
                "breakdown": [
                    {"part_type": "hood", "damage_type": "scratch", "cost": 1200, "hours": 1.5},
                ],
            }
        )

        await handle_confirm(event, {"rid": "req-1"}, backend, api)

        backend.confirm_pricing.assert_awaited_once_with("req-1")
        call_kwargs = api.messages.send.call_args.kwargs
        assert "5,000" in call_kwargs["message"] or "5000" in call_kwargs["message"]
        assert "3.5" in call_kwargs["message"]
        assert "Капот" in call_kwargs["message"]

    async def test_backend_error_sends_error_message(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.confirm_pricing = AsyncMock(side_effect=Exception("fail"))

        await handle_confirm(event, {"rid": "req-1"}, backend, api)

        call_kwargs = api.messages.send.call_args.kwargs
        assert "ошибка" in call_kwargs["message"].lower()
