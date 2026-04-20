"""QA: bot callback handlers must survive malformed payloads.

``event.get_payload_json()`` returns whatever the VK server delivered,
which in turn is whatever attacker/bot-client put into the ``payload``
field of a callback button. In steady state the bot emits its own
payloads with well-known keys (``rid``, ``did``, ``pt``, ``dt``, ``m``),
but:

* a stale button from an older bot version may carry a legacy payload
  without a ``rid`` key,
* a malicious user can trigger a callback with any JSON they want by
  crafting a keyboard in the VK client (vkontakte allows sending button
  payloads directly via the API),
* a corrupted ``get_payload_json`` returning ``{}`` is already handled
  in the dispatcher via ``or {}``.

In every case the handler has to degrade gracefully: log the error,
answer the user with a neutral message, and return. Raising ``KeyError``
would propagate through ``vkbottle`` and swallow the snackbar feedback,
leaving the user staring at a spinner.

These tests check the "graceful failure" contract for each callback
handler by feeding it an empty payload and asserting no uncaught
exception propagates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.damage_edit import (
    handle_edit_action,
    handle_edit_damage_type,
)
from auto_repair_estimator.bot.handlers.manual_flow import (
    handle_damage_type_selection,
    handle_mode_selection,
    handle_part_selection,
)
from auto_repair_estimator.bot.handlers.pricing import handle_confirm


def _event():
    event = MagicMock()
    event.peer_id = 100
    event.user_id = 42
    return event


def _api():
    api = AsyncMock()
    api.messages.send = AsyncMock()
    return api


def _backend():
    return AsyncMock(spec=BackendClient)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "handler,required_keys",
    [
        # Each handler's minimal required payload keys. When they're all
        # missing, the handler must not raise — it should send a generic
        # error message or just log-and-return.
        (handle_mode_selection, ("m",)),
        (handle_part_selection, ("pt", "rid")),
        (handle_damage_type_selection, ("rid", "pt", "dt")),
        (handle_edit_action, ("a", "rid")),
        (handle_edit_damage_type, ("rid", "did", "dt")),
        (handle_confirm, ("rid",)),
    ],
    ids=[
        "mode_selection",
        "part_selection",
        "damage_type_selection",
        "edit_action",
        "edit_damage_type",
        "confirm",
    ],
)
async def test_handler_does_not_raise_keyerror_on_missing_payload_keys(
    handler, required_keys: tuple[str, ...]
) -> None:
    """Feed an empty payload (every required key is missing) and assert
    the handler returns without bubbling a KeyError to the dispatcher.

    The dispatcher in ``bot/main.py`` has no outer ``try/except`` around
    handler calls, so a KeyError propagates through ``vkbottle`` and the
    user never receives a response. The handler itself must own this
    failure mode.
    """

    event = _event()
    api = _api()
    backend = _backend()
    # Ensure AsyncMock.methods don't raise by themselves when called.
    for attr in dir(backend):
        if not attr.startswith("_"):
            candidate = getattr(backend, attr)
            if isinstance(candidate, AsyncMock):
                candidate.return_value = {}

    try:
        await handler(event, {}, backend, api)
    except KeyError as exc:
        pytest.fail(
            f"{handler.__name__} raised KeyError({exc!r}) for an empty payload. "
            f"Expected keys: {required_keys}. The handler must degrade "
            "gracefully — a raised KeyError escapes the vkbottle dispatcher and "
            "the user never sees a reply."
        )
    except Exception as exc:  # pragma: no cover - surface any other crash too
        pytest.fail(
            f"{handler.__name__} raised {type(exc).__name__}({exc!r}) for an "
            "empty payload. Expected a user-facing error message instead."
        )


@pytest.mark.anyio
async def test_confirm_with_non_string_rid_does_not_crash_on_formatting() -> None:
    """Edge case: ``payload["rid"]`` is supposed to be a UUID string, but if
    a future caller ever passes an integer or None, the downstream HTTP
    client must receive a defensive cast or produce a user-visible error
    rather than crashing in f-string formatting."""

    event = _event()
    api = _api()
    backend = AsyncMock(spec=BackendClient)
    backend.confirm_pricing = AsyncMock(side_effect=TypeError("bad rid"))

    # Should produce the generic "Ошибка при расчёте стоимости..." message,
    # not propagate TypeError.
    await handle_confirm(event, {"rid": 12345}, backend, api)

    sent = api.messages.send.await_args_list
    assert sent, "Handler didn't respond to the user after a backend TypeError."
    assert any("ошибка" in call.kwargs.get("message", "").lower() for call in sent)
