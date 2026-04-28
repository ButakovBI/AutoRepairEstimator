"""Tests for the "Начать"-button onboarding and reject-outside-session flow.

We assert on the observable behavior of the new pieces:

* ``start_keyboard()`` produces exactly one ``cmd=start`` inline button.
* ``handle_start_callback`` behaves like ``/start`` (welcome + mode keyboard).
* ``send_no_active_session_reply`` always includes the start keyboard.
* ``BackendClient.get_active_request`` translates a 404 to ``None`` rather
  than raising -- the bot relies on that contract to skip an active session.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.start import (
    NO_ACTIVE_SESSION_TEXT,
    handle_start_callback,
    send_no_active_session_reply,
)
from auto_repair_estimator.bot.keyboards.start import start_keyboard


def _parse_keyboard(kb_json: str) -> dict:
    return json.loads(kb_json)


def _payload(btn: dict) -> dict:
    raw = btn["action"]["payload"]
    return json.loads(raw) if isinstance(raw, str) else raw


class TestStartKeyboard:
    def test_single_button_with_start_cmd(self):
        # The reject reply attaches this keyboard -- it must have exactly one
        # obvious "Начать" button so the user can't fail to spot it.
        kb = _parse_keyboard(start_keyboard())
        all_buttons = [btn for row in kb["buttons"] for btn in row]
        assert len(all_buttons) == 1
        payload = _payload(all_buttons[0])
        assert payload == {"cmd": "start"}

    def test_keyboard_is_inline(self):
        kb = _parse_keyboard(start_keyboard())
        assert kb["inline"] is True


class TestHandleStartCallback:
    async def test_sends_welcome_message_and_mode_keyboard(self):
        # cmd=start callback must yield the same onboarding experience as the
        # /start text command: welcome text + ML/Manual choice.
        event = MagicMock()
        event.peer_id = 123
        api = AsyncMock()
        api.messages.send = AsyncMock()
        # No prior active — plain welcome with no notice prefix.
        backend = AsyncMock(spec=BackendClient)
        backend.get_active_request = AsyncMock(return_value=None)
        backend.abandon_request = AsyncMock()

        await handle_start_callback(event, {"cmd": "start"}, backend, api)

        api.messages.send.assert_awaited_once()
        call_kwargs = api.messages.send.call_args.kwargs
        assert call_kwargs["peer_id"] == 123
        assert "Добро пожаловать" in call_kwargs["message"]
        kb = _parse_keyboard(call_kwargs["keyboard"])
        cmds = {_payload(b)["cmd"] for row in kb["buttons"] for b in row}
        assert cmds == {"mode"}


class TestSendNoActiveSessionReply:
    async def test_sends_reject_text_with_start_keyboard(self):
        api = AsyncMock()
        api.messages.send = AsyncMock()

        await send_no_active_session_reply(api, peer_id=777)

        api.messages.send.assert_awaited_once()
        call_kwargs = api.messages.send.call_args.kwargs
        assert call_kwargs["peer_id"] == 777
        assert call_kwargs["message"] == NO_ACTIVE_SESSION_TEXT
        # Keyboard must be the single-button "Начать" keyboard so the user
        # has a zero-typing way out of this state.
        kb = _parse_keyboard(call_kwargs["keyboard"])
        btns = [b for row in kb["buttons"] for b in row]
        assert len(btns) == 1
        assert _payload(btns[0]) == {"cmd": "start"}


class _FakeAsyncHttpClient:
    """Minimal httpx.AsyncClient stub that returns a preconfigured response."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.closed = False

    async def get(self, url: str, params: dict | None = None) -> httpx.Response:  # noqa: ARG002
        return self._response

    async def aclose(self) -> None:
        self.closed = True


class TestBackendClientGetActiveRequest:
    async def test_returns_none_on_404(self):
        # The 404 is the "no active session" signal -- it must NOT raise
        # (the bot's caller treats None as a control-flow branch, not an error).
        resp = httpx.Response(404, request=httpx.Request("GET", "http://x/v1/requests/active"))
        client = BackendClient("http://example")
        client._client = _FakeAsyncHttpClient(resp)  # type: ignore[assignment]  # noqa: SLF001

        result = await client.get_active_request(chat_id=1)

        assert result is None

    async def test_returns_parsed_json_on_200(self):
        payload: dict[str, Any] = {
            "id": "req-1",
            "status": "pricing",
            "mode": "manual",
            "chat_id": 1,
        }
        resp = httpx.Response(
            200,
            json=payload,
            request=httpx.Request("GET", "http://x/v1/requests/active"),
        )
        client = BackendClient("http://example")
        client._client = _FakeAsyncHttpClient(resp)  # type: ignore[assignment]  # noqa: SLF001

        result = await client.get_active_request(chat_id=1)

        assert result == payload

    async def test_raises_on_500(self):
        # Genuine server errors must surface -- silently swallowing them
        # would turn "backend is down" into "no active session", which would
        # reset every user's progress.
        resp = httpx.Response(500, request=httpx.Request("GET", "http://x/v1/requests/active"))
        client = BackendClient("http://example")
        client._client = _FakeAsyncHttpClient(resp)  # type: ignore[assignment]  # noqa: SLF001

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_active_request(chat_id=1)
