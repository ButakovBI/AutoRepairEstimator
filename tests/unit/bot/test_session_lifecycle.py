"""Tests for the single-active-session invariant.

This suite pins the user-facing behaviour of the session-lifecycle
round:

* **Multiple photos in one VK message** → only the first is processed,
  the others are explicitly acknowledged as ignored.
* **Photo arriving during any active session** (ML at any status or
  manual) → the prior session is abandoned, the user sees the shared
  ``PREVIOUS_REQUEST_ABANDONED_NOTICE``, the new ML request is created.
* **"Начать" (text and callback) mid-scenario** → prior session is
  abandoned, welcome text is prefixed with the notice, mode keyboard
  is shown.
* **Shared helper** ``abandon_active_session`` is idempotent and
  resilient to backend hiccups (does not block the new-session path).

The backend contracts touched here (``get_active_request`` →
``abandon_request``) are exercised via mocks; the real backend use
cases are covered separately.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.photo import (
    MULTIPLE_PHOTOS_NOTICE,
    handle_photo,
)
from auto_repair_estimator.bot.handlers.start import (
    handle_start,
    handle_start_callback,
)
from auto_repair_estimator.bot.session_lifecycle import (
    PREVIOUS_REQUEST_ABANDONED_NOTICE,
    abandon_active_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(
    *,
    active: dict | None = None,
    active_error: Exception | None = None,
    abandon_error: Exception | None = None,
    create: dict | None = None,
) -> AsyncMock:
    b = AsyncMock(spec=BackendClient)
    if active_error is not None:
        b.get_active_request = AsyncMock(side_effect=active_error)
    else:
        b.get_active_request = AsyncMock(return_value=active)
    if abandon_error is not None:
        b.abandon_request = AsyncMock(side_effect=abandon_error)
    else:
        b.abandon_request = AsyncMock(
            return_value={"id": "prev", "status": "failed"}
        )
    if create is not None:
        b.create_request = AsyncMock(return_value=create)
    else:
        b.create_request = AsyncMock(return_value={"id": "new-rid", "presigned_put_url": None})
    b.upload_photo = AsyncMock()
    return b


def _photo_attachment(url: str = "https://vk.test/p.jpg") -> MagicMock:
    size = MagicMock()
    size.width = 800
    size.height = 600
    size.url = url
    ph = MagicMock()
    ph.sizes = [size]
    return ph


def _message(peer_id: int = 10, photos: list | None = None) -> MagicMock:
    msg = MagicMock()
    msg.peer_id = peer_id
    msg.from_id = 20
    msg.conversation_message_id = 777
    msg.get_photo_attachments = MagicMock(return_value=photos or [])
    msg.answer = AsyncMock()
    return msg


class _FakeHttpx:
    """Minimal httpx.AsyncClient stand-in for the photo handler tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        get_resp = MagicMock()
        get_resp.content = b"fake-jpeg"
        get_resp.raise_for_status = MagicMock()
        put_resp = MagicMock()
        put_resp.raise_for_status = MagicMock()
        self.get = AsyncMock(return_value=get_resp)
        self.put = AsyncMock(return_value=put_resp)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# abandon_active_session (the shared primitive)
# ---------------------------------------------------------------------------


class TestAbandonActiveSession:
    """Contract tests for the shared helper every entry point depends on."""

    async def test_returns_active_dict_when_it_closed_one(self):
        active = {"id": "r-1", "mode": "ml", "status": "processing"}
        b = _backend(active=active)

        result = await abandon_active_session(b, chat_id=42)

        # The probe then the abandon — in that order.
        b.get_active_request.assert_awaited_once_with(42)
        b.abandon_request.assert_awaited_once_with("r-1")
        # The caller needs the active dict to decide whether to render
        # the "previous session closed" notice; we return the one the
        # backend surfaced BEFORE the abandon call (so mode/status stay
        # readable for contextual messages).
        assert result == active

    async def test_returns_none_when_chat_has_no_active_session(self):
        b = _backend(active=None)

        result = await abandon_active_session(b, chat_id=42)

        assert result is None
        # No wasted abandon round-trip in the common case.
        b.abandon_request.assert_not_awaited()

    async def test_probe_failure_degrades_silently(self):
        # A 5xx on the probe must NOT prevent the caller from creating
        # the new session. The invariant temporarily loosens to "at
        # most one within 5 minutes" while the watchdog catches up.
        b = _backend(active_error=RuntimeError("500"))

        result = await abandon_active_session(b, chat_id=42)

        assert result is None
        b.abandon_request.assert_not_awaited()

    async def test_abandon_failure_still_reports_intent(self):
        # If the abandon HTTP call itself fails (network blip, backend
        # restart), we still return the active dict — otherwise the
        # caller would render a "nothing changed" UI even though we
        # tried. The watchdog converges DB state within 5 minutes.
        active = {"id": "r-1", "mode": "ml", "status": "created"}
        b = _backend(active=active, abandon_error=RuntimeError("503"))

        result = await abandon_active_session(b, chat_id=42)

        assert result == active

    async def test_active_without_id_is_ignored(self):
        # Defensive: a malformed "active" payload (no id) would make us
        # call abandon_request("None") which would 400 on the backend.
        # We treat it as "no session to close".
        b = _backend(active={"id": None, "mode": "ml", "status": "created"})

        result = await abandon_active_session(b, chat_id=42)

        assert result is None
        b.abandon_request.assert_not_awaited()


# ---------------------------------------------------------------------------
# handle_photo — single-photo + abandon + notice
# ---------------------------------------------------------------------------


class TestHandlePhotoAbandonPrior:
    async def test_prior_active_is_abandoned_and_user_is_notified(self):
        b = _backend(
            active={"id": "old", "mode": "ml", "status": "processing"},
            create={"id": "new-rid", "presigned_put_url": "https://minio/put?x=1"},
        )
        msg = _message(photos=[_photo_attachment()])
        api = MagicMock()

        with patch(
            "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
            _FakeHttpx,
        ):
            await handle_photo(msg, b, api)

        b.abandon_request.assert_awaited_once_with("old")
        b.create_request.assert_awaited_once()
        # The first assistant message must carry the "previous closed"
        # notice so the user doesn't lose context silently.
        intro = msg.answer.await_args_list[0].args[0]
        assert PREVIOUS_REQUEST_ABANDONED_NOTICE in intro

    async def test_no_prior_active_means_no_abandon_call(self):
        b = _backend(
            active=None,
            create={"id": "new-rid", "presigned_put_url": "https://minio/put?x=1"},
        )
        msg = _message(photos=[_photo_attachment()])
        api = MagicMock()

        with patch(
            "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
            _FakeHttpx,
        ):
            await handle_photo(msg, b, api)

        b.abandon_request.assert_not_awaited()
        intro = msg.answer.await_args_list[0].args[0]
        assert PREVIOUS_REQUEST_ABANDONED_NOTICE not in intro

    async def test_photo_during_manual_session_abandons_and_proceeds(self):
        # Regression: the old behaviour hard-rejected photos during
        # manual flows; the new contract replaces them. The user who
        # gave up on manual entry and sent a photo gets it processed.
        b = _backend(
            active={"id": "old", "mode": "manual", "status": "pricing"},
            create={"id": "new-rid", "presigned_put_url": "https://minio/put?x=1"},
        )
        msg = _message(photos=[_photo_attachment()])
        api = MagicMock()

        with patch(
            "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
            _FakeHttpx,
        ):
            await handle_photo(msg, b, api)

        b.abandon_request.assert_awaited_once_with("old")
        b.create_request.assert_awaited_once()
        # ``mode="ml"`` in create_request args — the manual session is
        # gone and the replacement is an ML one (photos only go to ML).
        create_kwargs = b.create_request.await_args.kwargs
        assert create_kwargs["mode"] == "ml"


class TestHandlePhotoMultipleAttachments:
    async def test_multiple_photos_create_exactly_one_request(self):
        # Core invariant: N attachments → 1 RepairRequest. The old
        # "one-per-photo" path created parallel non-terminal rows for
        # the same chat_id and made the bot unpredictable — not allowed.
        b = _backend(
            active=None,
            create={"id": "new-rid", "presigned_put_url": "https://minio/put?x=1"},
        )
        msg = _message(
            photos=[
                _photo_attachment("https://vk.test/a.jpg"),
                _photo_attachment("https://vk.test/b.jpg"),
                _photo_attachment("https://vk.test/c.jpg"),
            ]
        )
        api = MagicMock()

        with patch(
            "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
            _FakeHttpx,
        ):
            await handle_photo(msg, b, api)

        assert b.create_request.await_count == 1

    async def test_multiple_photos_notice_is_sent_to_user(self):
        b = _backend(
            active=None,
            create={"id": "new-rid", "presigned_put_url": "https://minio/put?x=1"},
        )
        msg = _message(
            photos=[_photo_attachment("https://vk.test/a.jpg"), _photo_attachment("https://vk.test/b.jpg")]
        )
        api = MagicMock()

        with patch(
            "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
            _FakeHttpx,
        ):
            await handle_photo(msg, b, api)

        intro = msg.answer.await_args_list[0].args[0]
        # The exact user-facing string — pin it so future rewordings
        # show up as a visible test diff, not a silent regression.
        assert MULTIPLE_PHOTOS_NOTICE in intro

    async def test_single_photo_does_not_emit_multiple_notice(self):
        b = _backend(
            active=None,
            create={"id": "new-rid", "presigned_put_url": "https://minio/put?x=1"},
        )
        msg = _message(photos=[_photo_attachment()])
        api = MagicMock()

        with patch(
            "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
            _FakeHttpx,
        ):
            await handle_photo(msg, b, api)

        intro = msg.answer.await_args_list[0].args[0]
        assert MULTIPLE_PHOTOS_NOTICE not in intro


# ---------------------------------------------------------------------------
# handle_start (/start / "Начать" text)
# ---------------------------------------------------------------------------


def _parse_kb(kb: str) -> dict:
    return json.loads(kb)


def _button_cmds(kb: str) -> set[str]:
    data = _parse_kb(kb)
    cmds: set[str] = set()
    for row in data["buttons"]:
        for btn in row:
            raw = btn["action"]["payload"]
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if c := payload.get("cmd"):
                cmds.add(c)
    return cmds


class TestHandleStartTextCommand:
    async def test_prior_active_is_abandoned_and_welcome_prefixed(self):
        msg = _message()
        msg.peer_id = 777
        b = _backend(active={"id": "old", "mode": "manual", "status": "pricing"})

        await handle_start(msg, b)

        b.abandon_request.assert_awaited_once_with("old")
        call_kwargs = msg.answer.await_args.kwargs
        text = call_kwargs["message"]
        assert text.startswith(PREVIOUS_REQUEST_ABANDONED_NOTICE)
        # Mode keyboard is still attached — the user lands on the
        # "choose ML / manual" screen as if they'd pressed Начать fresh.
        assert _button_cmds(call_kwargs["keyboard"]) == {"mode"}

    async def test_no_prior_active_sends_plain_welcome(self):
        msg = _message(peer_id=777)
        b = _backend(active=None)

        await handle_start(msg, b)

        b.abandon_request.assert_not_awaited()
        text = msg.answer.await_args.kwargs["message"]
        assert not text.startswith(PREVIOUS_REQUEST_ABANDONED_NOTICE)


class TestHandleStartCallback:
    async def test_prior_active_is_abandoned_and_welcome_prefixed(self):
        event = MagicMock()
        event.peer_id = 555
        api = AsyncMock()
        api.messages.send = AsyncMock()
        b = _backend(active={"id": "old", "mode": "ml", "status": "processing"})

        await handle_start_callback(event, {"cmd": "start"}, b, api)

        b.abandon_request.assert_awaited_once_with("old")
        call_kwargs = api.messages.send.call_args.kwargs
        text = call_kwargs["message"]
        assert text.startswith(PREVIOUS_REQUEST_ABANDONED_NOTICE)
        assert _button_cmds(call_kwargs["keyboard"]) == {"mode"}

    async def test_no_prior_active_is_plain_welcome(self):
        event = MagicMock()
        event.peer_id = 555
        api = AsyncMock()
        api.messages.send = AsyncMock()
        b = _backend(active=None)

        await handle_start_callback(event, {"cmd": "start"}, b, api)

        text = api.messages.send.call_args.kwargs["message"]
        assert not text.startswith(PREVIOUS_REQUEST_ABANDONED_NOTICE)
