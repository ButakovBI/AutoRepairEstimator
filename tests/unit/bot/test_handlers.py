"""Behavioral tests for VK bot handler functions.

Tests verify that each handler:
- Calls the correct backend methods with the right arguments
- Sends the appropriate VK messages via the API
- Handles error cases gracefully
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

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
from auto_repair_estimator.bot.handlers.photo import handle_photo
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


def _make_photo_attachment(url: str) -> MagicMock:
    size = MagicMock()
    size.width = 800
    size.height = 600
    size.url = url
    photo = MagicMock()
    photo.sizes = [size]
    return photo


class _FakeHttpxAsyncClient:
    """Minimal async context manager replacing httpx.AsyncClient in photo handler tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._get_response = MagicMock()
        self._get_response.content = b"fake-jpeg"
        self._get_response.raise_for_status = MagicMock()
        self._put_response = MagicMock()
        self._put_response.raise_for_status = MagicMock()
        self.get = AsyncMock(return_value=self._get_response)
        self.put = AsyncMock(return_value=self._put_response)

    async def __aenter__(self) -> _FakeHttpxAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False


class TestHandlePhoto:
    async def test_single_photo_does_not_send_multi_attachment_notice(self):
        message = MagicMock()
        message.peer_id = 10
        message.from_id = 20
        message.get_photo_attachments = MagicMock(return_value=[_make_photo_attachment("https://vk.test/a.jpg")])
        message.answer = AsyncMock()
        backend = _make_backend(create_request={"id": "req-1", "presigned_put_url": None})
        api = MagicMock()

        with patch("auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient", _FakeHttpxAsyncClient):
            await handle_photo(message, backend, api)

        assert message.answer.await_count == 2
        first_text = message.answer.await_args_list[0].args[0]
        assert "несколько" not in first_text.lower()
        assert "Загружаю" in first_text

    async def test_multiple_photos_create_one_request_per_photo(self):
        # Arrange — two photos in one VK message
        message = MagicMock()
        message.peer_id = 10
        message.from_id = 20
        message.get_photo_attachments = MagicMock(
            return_value=[
                _make_photo_attachment("https://vk.test/first.jpg"),
                _make_photo_attachment("https://vk.test/second.jpg"),
            ]
        )
        message.answer = AsyncMock()
        backend = AsyncMock(spec=BackendClient)
        backend.create_request = AsyncMock(
            side_effect=[
                {"id": "req-1", "presigned_put_url": None},
                {"id": "req-2", "presigned_put_url": None},
            ]
        )
        backend.upload_photo = AsyncMock()
        api = MagicMock()

        # Act
        with patch("auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient", _FakeHttpxAsyncClient):
            await handle_photo(message, backend, api)

        # Assert — one backend request per photo, each with its own upload
        assert backend.create_request.await_count == 2
        assert backend.upload_photo.await_count == 2
        first_upload = backend.upload_photo.await_args_list[0].kwargs
        second_upload = backend.upload_photo.await_args_list[1].kwargs
        assert first_upload["request_id"] == "req-1"
        assert second_upload["request_id"] == "req-2"
        assert first_upload["image_key"] == "raw-images/req-1.jpg"
        assert second_upload["image_key"] == "raw-images/req-2.jpg"

        # Intro mentions batch handling
        intro = message.answer.await_args_list[0].args[0]
        assert "2" in intro

    async def test_photo_upload_uses_presigned_url_when_provided(self):
        # Arrange
        message = MagicMock()
        message.peer_id = 10
        message.from_id = 20
        message.get_photo_attachments = MagicMock(return_value=[_make_photo_attachment("https://vk.test/a.jpg")])
        message.answer = AsyncMock()
        backend = _make_backend(
            create_request={"id": "req-1", "presigned_put_url": "https://minio/put?sig=1"}
        )
        api = MagicMock()

        clients: list[_FakeHttpxAsyncClient] = []

        def _client_factory(*args: object, **kwargs: object) -> _FakeHttpxAsyncClient:
            client = _FakeHttpxAsyncClient()
            clients.append(client)
            return client

        # Act
        with patch("auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient", side_effect=_client_factory):
            await handle_photo(message, backend, api)

        # Assert — a PUT to the presigned URL actually occurred
        puts = [c.put for c in clients if c.put.await_count > 0]
        assert puts, "presigned PUT must be called exactly once"
        assert puts[0].await_args.args[0] == "https://minio/put?sig=1"


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
        assert len(all_btns) == 12

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
        assert len(all_btns) == 8


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
        assert len(all_btns) == 12


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
        # Hood × scratch per thesis table 5: 10-18 тыс. руб. / 8 h painting.
        backend = _make_backend(
            confirm_pricing={
                "total_cost_min": 10_000.0,
                "total_cost_max": 18_000.0,
                "total_hours_min": 8.0,
                "total_hours_max": 8.0,
                "breakdown": [
                    {
                        "part_type": "hood",
                        "damage_type": "scratch",
                        "cost_min": 10_000.0,
                        "cost_max": 18_000.0,
                        "hours_min": 8.0,
                        "hours_max": 8.0,
                    },
                ],
                "notes": [],
            }
        )

        await handle_confirm(event, {"rid": "req-1"}, backend, api)

        backend.confirm_pricing.assert_awaited_once_with("req-1")
        call_kwargs = api.messages.send.call_args.kwargs
        message = call_kwargs["message"]
        # Range must be rendered as "10,000–18,000 руб." (en-dash U+2013).
        assert "10,000\u201318,000" in message
        assert "8 ч" in message
        assert "Капот" in message

    async def test_backend_error_sends_error_message(self):
        event = _make_event()
        api = _make_api()
        backend = AsyncMock(spec=BackendClient)
        backend.confirm_pricing = AsyncMock(side_effect=Exception("fail"))

        await handle_confirm(event, {"rid": "req-1"}, backend, api)

        call_kwargs = api.messages.send.call_args.kwargs
        assert "ошибка" in call_kwargs["message"].lower()

    async def test_wheel_only_request_shows_tyre_shop_note_and_no_price(self):
        """Backend returns zero totals, empty breakdown and a tyre-shop note
        when every damage was on a wheel. The bot must render the note and
        must not report any nonsense numeric estimate."""
        event = _make_event()
        api = _make_api()
        backend = _make_backend(
            confirm_pricing={
                "total_cost_min": 0.0,
                "total_cost_max": 0.0,
                "total_hours_min": 0.0,
                "total_hours_max": 0.0,
                "breakdown": [],
                "notes": ["Для повреждений шин и колёсных дисков кузовной ремонт не применим — обратитесь в шиномонтаж."],
            }
        )

        await handle_confirm(event, {"rid": "req-1"}, backend, api)

        message = api.messages.send.call_args.kwargs["message"]
        assert "шиномонтаж" in message.lower()
        # No 0 руб. / 0 ч line should be shown — we explicitly branch to the
        # "кузовной ремонт не требуется" copy when breakdown is empty.
        assert "Стоимость ремонта" not in message
        assert "кузовной ремонт" in message.lower()

    async def test_range_and_polish_note_are_both_rendered_for_scratch(self):
        event = _make_event()
        api = _make_api()
        # Door × scratch: 10-18 тыс / 8 h painting; plus polish note.
        backend = _make_backend(
            confirm_pricing={
                "total_cost_min": 10_000.0,
                "total_cost_max": 18_000.0,
                "total_hours_min": 8.0,
                "total_hours_max": 8.0,
                "breakdown": [
                    {
                        "part_type": "door",
                        "damage_type": "scratch",
                        "cost_min": 10_000.0,
                        "cost_max": 18_000.0,
                        "hours_min": 8.0,
                        "hours_max": 8.0,
                    }
                ],
                "notes": ["Если достаточно полировки — 1 ч и 1,000 руб."],
            }
        )

        await handle_confirm(event, {"rid": "req-1"}, backend, api)

        message = api.messages.send.call_args.kwargs["message"]
        # En-dash formatting for the cost range.
        assert "10,000\u201318,000 руб." in message
        # Exact-value hours render without a range (min==max).
        assert "8 ч" in message
        assert "полировки" in message.lower()
