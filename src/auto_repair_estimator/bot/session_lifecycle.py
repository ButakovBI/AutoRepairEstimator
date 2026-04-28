"""Shared helpers for enforcing the "one active session per chat" invariant.

The bot has four entry points that can create a fresh ``RepairRequest``:

* ``handle_photo`` — a photo arrived,
* ``handle_mode_selection`` — the user picked ML or Manual,
* ``handle_start`` — the user typed ``/start`` / «Начать»,
* ``handle_start_callback`` — the user pressed the inline "Начать" button.

Previously each of them had its own story about what happens if the chat
already has a non-terminal session. That inconsistency produced the UX
issues the user reported: spamming photos created a fan-out of parallel
ML requests, pressing "Начать" left the old session quietly alive, and
sending a photo inside a manual flow rejected the photo outright. This
module gives all four paths the same primitive — ``abandon_active_session``
— plus a shared notice string so the user is always told, in identical
wording, that their previous scenario was closed.

Design notes:

* The helper is deliberately silent on the "nothing to abandon" branch —
  callers only need to render the notice when :meth:`abandon_active_session`
  returns a dict. That keeps the callsites free of conditional-string
  boilerplate.
* Every failure mode (backend 5xx, abandon endpoint unreachable, VK API
  hiccup) degrades to "pretend nothing was there". The worst case is a
  brief overlap where two non-terminal rows coexist for 5 minutes —
  the backend watchdog collapses that back to the invariant. We never
  block a user's new request on a degraded abandon.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from auto_repair_estimator.bot.backend_client import BackendClient

PREVIOUS_REQUEST_ABANDONED_NOTICE = "Предыдущая заявка закрыта."


async def abandon_active_session(
    backend: BackendClient, chat_id: int
) -> dict[str, Any] | None:
    """Close whatever non-terminal session the chat currently has.

    Returns the abandoned session's dict (as the backend reported it
    before the abandon call) if a session was actually closed, else
    ``None``. Callers use the return value to decide whether to render
    the ``PREVIOUS_REQUEST_ABANDONED_NOTICE`` to the user.

    The function is idempotent: calling it when the backend has no
    active session, or when the active session is already terminal,
    both return ``None`` without raising. A 5xx on the probe or the
    abandon call is logged at WARNING and swallowed — the user's new
    request must not be blocked by an unrelated backend hiccup.
    """

    try:
        active = await backend.get_active_request(chat_id)
    except Exception as exc:
        logger.warning(
            "Could not probe active request for chat_id={} before abandon: {}",
            chat_id,
            exc,
        )
        return None
    if active is None:
        return None

    rid = active.get("id")
    if not rid:
        logger.warning(
            "Active session for chat_id={} has no id field: {}", chat_id, active
        )
        return None

    try:
        await backend.abandon_request(str(rid))
    except Exception as exc:
        logger.warning(
            "abandon_request failed for chat_id={} rid={}: {}", chat_id, rid, exc
        )
        # Even if abandon failed, we still report the intent so the
        # caller UX matches what the user expects ("I pressed Начать,
        # my old thing should be gone"). The watchdog will converge
        # the DB state within 5 minutes.
    return active


__all__ = [
    "PREVIOUS_REQUEST_ABANDONED_NOTICE",
    "abandon_active_session",
]
