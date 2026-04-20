"""Behavioural tests for the top-level bot dispatch helpers.

``handle_incoming_message`` and ``validate_active_rid_for_callback`` are
the two decision points that used to live as closures inside
``_register_handlers``. They encode every bug-fix from the latest UX
round (bugs #2/#3/#4) so we assert each branch with explicit fakes:

* photo + active MANUAL → photo-gating nudge, no ML request
* photo + active ML / no active → delegates to ``handle_photo``
* text + active → ``active_session_nudge`` is delivered (never the
  generic "use the buttons" copy when the last message had none)
* callback validation: missing rid, missing active, rid mismatch,
  manual-flow cmd outside PRICING — each path ends in a single rejection
  nudge, never in a backend mutation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from auto_repair_estimator.bot.handlers.start import (
    NO_ACTIVE_SESSION_TEXT,
    PHOTO_DURING_MANUAL_NUDGE,
)
from auto_repair_estimator.bot.main import (
    CALLBACK_HANDLERS,
    CMDS_REQUIRING_ACTIVE_RID,
    CMDS_REQUIRING_PRICING_STATUS,
    handle_incoming_message,
    validate_active_rid_for_callback,
)


def _fake_backend(active: dict[str, Any] | None = None, *, raises: Exception | None = None) -> MagicMock:
    backend = MagicMock()
    if raises is not None:
        backend.get_active_request = AsyncMock(side_effect=raises)
    else:
        backend.get_active_request = AsyncMock(return_value=active)
    return backend


def _fake_api() -> MagicMock:
    api = MagicMock()
    api.messages = MagicMock()
    api.messages.send = AsyncMock()
    return api


def _fake_message(peer_id: int = 42, photos: list | None = None) -> MagicMock:
    msg = MagicMock()
    msg.peer_id = peer_id
    msg.get_photo_attachments = MagicMock(return_value=photos or [])
    msg.answer = AsyncMock()
    return msg


def _sent_texts(api: MagicMock) -> list[str]:
    """Collect every ``message=`` kwarg sent through the API mock."""
    return [call.kwargs.get("message") for call in api.messages.send.await_args_list]


class TestHandleIncomingMessagePhotoPath:
    """Photo-branch invariants from bug #4 and the broader photo pipeline."""

    async def test_photo_during_manual_is_gated(self):
        # Active MANUAL session + photo arrives → we must nudge the user
        # to switch modes instead of silently creating a parallel ML
        # request (which would leave two non-terminal rows per chat).
        backend = _fake_backend(active={"id": "r-1", "mode": "manual", "status": "pricing"})
        api = _fake_api()
        msg = _fake_message(photos=[MagicMock()])

        with patch("auto_repair_estimator.bot.main.handle_photo", new=AsyncMock()) as fake_photo:
            await handle_incoming_message(backend, api, msg)
            fake_photo.assert_not_awaited()

        msg.answer.assert_awaited_once()
        assert msg.answer.await_args.args[0] == PHOTO_DURING_MANUAL_NUDGE

    async def test_photo_with_no_active_session_goes_to_ml(self):
        # No active → photo is the start of a brand-new ML request. The
        # "no active session" reject must NOT fire on this path.
        backend = _fake_backend(active=None)
        api = _fake_api()
        msg = _fake_message(photos=[MagicMock()])

        with patch("auto_repair_estimator.bot.main.handle_photo", new=AsyncMock()) as fake_photo:
            await handle_incoming_message(backend, api, msg)
            fake_photo.assert_awaited_once_with(msg, backend, api)

        msg.answer.assert_not_awaited()
        api.messages.send.assert_not_awaited()

    async def test_photo_with_active_ml_session_goes_to_ml(self):
        # ML already active + another photo is a legitimate "retry with
        # a clearer shot". handle_photo handles dedup/abandon logic.
        backend = _fake_backend(active={"id": "r-1", "mode": "ml", "status": "processing"})
        api = _fake_api()
        msg = _fake_message(photos=[MagicMock()])

        with patch("auto_repair_estimator.bot.main.handle_photo", new=AsyncMock()) as fake_photo:
            await handle_incoming_message(backend, api, msg)
            fake_photo.assert_awaited_once()

    async def test_backend_probe_failure_does_not_block_photo(self):
        # Probe raised → we degrade to "active unknown" and still take the
        # non-manual branch so a transient backend hiccup never swallows
        # the user's photo upload.
        backend = _fake_backend(raises=RuntimeError("boom"))
        api = _fake_api()
        msg = _fake_message(photos=[MagicMock()])

        with patch("auto_repair_estimator.bot.main.handle_photo", new=AsyncMock()) as fake_photo:
            await handle_incoming_message(backend, api, msg)
            fake_photo.assert_awaited_once()


class TestHandleIncomingMessageTextPath:
    """Text-branch invariants from bug #2."""

    async def test_no_active_session_sends_reject(self):
        backend = _fake_backend(active=None)
        api = _fake_api()
        msg = _fake_message(photos=[])

        await handle_incoming_message(backend, api, msg)

        api.messages.send.assert_awaited_once()
        assert api.messages.send.await_args.kwargs["message"] == NO_ACTIVE_SESSION_TEXT
        msg.answer.assert_not_awaited()

    async def test_active_ml_awaiting_photo_gets_photo_nudge(self):
        # The classic bug #2 reproducer: before the fix, users staring at
        # "Отправьте чёткую фотографию..." who typed "ok?" got told to
        # "use the buttons" on a message that had zero buttons.
        backend = _fake_backend(active={"id": "r", "mode": "ml", "status": "created"})
        api = _fake_api()
        msg = _fake_message(photos=[])

        await handle_incoming_message(backend, api, msg)

        msg.answer.assert_awaited_once()
        text = msg.answer.await_args.args[0]
        assert "Жду фотографию" in text
        assert "Используйте кнопки" not in text

    async def test_active_manual_gets_manual_nudge(self):
        backend = _fake_backend(active={"id": "r", "mode": "manual", "status": "pricing"})
        api = _fake_api()
        msg = _fake_message(photos=[])

        await handle_incoming_message(backend, api, msg)

        msg.answer.assert_awaited_once()


class TestValidateActiveRidForCallback:
    """Callback-dispatch invariants from bug #3."""

    async def test_missing_rid_rejects_with_reject_copy(self):
        backend = _fake_backend(active={"id": "r-1", "status": "pricing", "mode": "manual"})
        api = _fake_api()

        ok = await validate_active_rid_for_callback(
            backend, api, peer_id=1, payload={"cmd": "part"}, cmd="part"
        )

        assert ok is False
        # ``get_active_request`` is irrelevant here — short-circuit on
        # missing rid avoids an unnecessary backend round-trip.
        backend.get_active_request.assert_not_awaited()
        assert api.messages.send.await_args.kwargs["message"] == NO_ACTIVE_SESSION_TEXT

    async def test_no_active_session_rejects(self):
        backend = _fake_backend(active=None)
        api = _fake_api()

        ok = await validate_active_rid_for_callback(
            backend, api, peer_id=1, payload={"cmd": "part", "rid": "r-1"}, cmd="part"
        )

        assert ok is False
        assert api.messages.send.await_args.kwargs["message"] == NO_ACTIVE_SESSION_TEXT

    async def test_rid_mismatch_rejects_without_mutating(self):
        # Stale button from a terminated session: payload says "r-old",
        # backend has "r-new". Dispatching would mutate the wrong row.
        backend = _fake_backend(active={"id": "r-new", "status": "pricing", "mode": "manual"})
        api = _fake_api()

        ok = await validate_active_rid_for_callback(
            backend, api, peer_id=1, payload={"cmd": "part", "rid": "r-old"}, cmd="part"
        )

        assert ok is False
        assert api.messages.send.await_args.kwargs["message"] == NO_ACTIVE_SESSION_TEXT

    async def test_manual_cmd_on_non_pricing_status_sends_mode_aware_nudge(self):
        # Every command in CMDS_REQUIRING_PRICING_STATUS must reject
        # cleanly (no backend mutation) when the session is still
        # PROCESSING — and the rejection copy should reflect the ML
        # state, not the generic error text.
        for cmd in CMDS_REQUIRING_PRICING_STATUS:
            backend = _fake_backend(active={"id": "r-1", "status": "processing", "mode": "ml"})
            api = _fake_api()

            ok = await validate_active_rid_for_callback(
                backend, api, peer_id=1, payload={"cmd": cmd, "rid": "r-1"}, cmd=cmd
            )

            assert ok is False, cmd
            text = api.messages.send.await_args.kwargs["message"]
            # ML/PROCESSING → "Ваш снимок уже обрабатывается"
            assert "обрабатывается" in text, cmd

    async def test_pricing_status_cmd_passes(self):
        backend = _fake_backend(active={"id": "r-1", "status": "pricing", "mode": "manual"})
        api = _fake_api()

        ok = await validate_active_rid_for_callback(
            backend, api, peer_id=1, payload={"cmd": "part", "rid": "r-1"}, cmd="part"
        )

        assert ok is True
        api.messages.send.assert_not_awaited()

    async def test_edit_cmd_does_not_require_pricing_status(self):
        # ``edit`` / ``edtype`` / ``confirm`` / ``back_edit`` sit on the
        # ML-edit side of the flow and must pass regardless of status
        # (the backend enforces the real state-machine guard). We only
        # gate the *manual* mutators — make sure we didn't over-reject.
        backend = _fake_backend(active={"id": "r-1", "status": "pricing", "mode": "ml"})
        api = _fake_api()

        ok = await validate_active_rid_for_callback(
            backend, api, peer_id=1, payload={"cmd": "edit", "rid": "r-1"}, cmd="edit"
        )

        assert ok is True

    async def test_backend_probe_failure_rejects(self):
        # If we can't even verify the active session, it's safer to
        # reject with the "press Начать" nudge than to dispatch blindly.
        backend = _fake_backend(raises=RuntimeError("kaput"))
        api = _fake_api()

        ok = await validate_active_rid_for_callback(
            backend, api, peer_id=1, payload={"cmd": "part", "rid": "r-1"}, cmd="part"
        )

        assert ok is False
        assert api.messages.send.await_args.kwargs["message"] == NO_ACTIVE_SESSION_TEXT


class TestCallbackPolicyTables:
    """Static invariants on the dispatch tables — cheap regression fence."""

    def test_every_rid_required_cmd_has_a_handler(self):
        missing = CMDS_REQUIRING_ACTIVE_RID - set(CALLBACK_HANDLERS)
        assert not missing, f"cmds gated on rid but not dispatched: {missing}"

    def test_pricing_required_cmds_are_subset_of_rid_required(self):
        # Logical invariant: "requires pricing" implies "requires rid".
        # If this ever fails, a cmd could bypass rid validation while
        # still demanding a pricing status → undefined behaviour.
        assert CMDS_REQUIRING_PRICING_STATUS <= CMDS_REQUIRING_ACTIVE_RID

    def test_start_and_mode_are_not_rid_gated(self):
        # ``start`` has no rid (it's how you create one) and ``mode``
        # creates the rid. Gating either would make onboarding impossible.
        assert "start" not in CMDS_REQUIRING_ACTIVE_RID
        assert "mode" not in CMDS_REQUIRING_ACTIVE_RID
