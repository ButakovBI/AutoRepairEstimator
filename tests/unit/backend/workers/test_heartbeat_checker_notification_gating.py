"""Behavioural tests for HeartbeatChecker notification gating.

Covers the two product rules introduced together:

* The user-facing timeout notification is only emitted for ML-mode
  requests. MANUAL-mode sessions don't involve asynchronous processing,
  so a watchdog sweep hitting one should *still* transition it to
  FAILED (hygienic cleanup of the DB) but must **not** push a user
  message — there's nothing for the user to retry that a generic
  "timed out" prompt would help with.
* Even an ML timeout is suppressed if the same chat already has another
  non-terminal request. That other request represents the user's
  current intent; shouting about the old one would confuse.

All tests work on the in-memory repo + in-memory outbox to keep the
contract tight against the real ``get_latest_active_by_chat_id`` query.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_outbox_repository import (
    InMemoryOutboxRepository,
)
from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    RequestMode,
    RequestStatus,
)
from auto_repair_estimator.backend.workers.heartbeat_checker import HeartbeatChecker


def _make_request(
    *,
    req_id: str,
    chat_id: int,
    mode: RequestMode,
    status: RequestStatus,
    timeout_at: datetime,
    created_at: datetime | None = None,
) -> RepairRequest:
    c = created_at or (datetime.now(UTC) - timedelta(minutes=10))
    return RepairRequest(
        id=req_id,
        chat_id=chat_id,
        user_id=chat_id + 1,  # unimportant; just not None
        mode=mode,
        status=status,
        created_at=c,
        updated_at=c,
        timeout_at=timeout_at,
    )


async def _run_checker() -> tuple[
    HeartbeatChecker,
    InMemoryRepairRequestRepository,
    InMemoryOutboxRepository,
]:
    requests = InMemoryRepairRequestRepository()
    outbox = InMemoryOutboxRepository()
    checker = HeartbeatChecker(
        request_repository=requests,
        outbox_repository=outbox,
        state_machine=RequestStateMachine(),
        notifications_topic="notifications",
    )
    return checker, requests, outbox


@pytest.mark.anyio
async def test_manual_timed_out_request_is_failed_but_no_notification_emitted() -> None:
    """MANUAL mode never yields a user-facing timeout message.

    Manual sessions don't have async processing — a "your request timed
    out, try again" pop-up has no useful meaning for the user.
    """

    checker, requests, outbox = await _run_checker()
    expired = _make_request(
        req_id="manual-expired",
        chat_id=1001,
        mode=RequestMode.MANUAL,
        status=RequestStatus.PRICING,
        timeout_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    await requests.add(expired)

    await checker._check_timeouts()

    # DB cleanup still happens:
    stored = await requests.get(expired.id)
    assert stored is not None
    assert stored.status is RequestStatus.FAILED
    # But no outbox event — the bot never gets a notify for manual sessions.
    assert await outbox.get_unpublished(limit=10) == []


@pytest.mark.anyio
async def test_ml_timeout_is_suppressed_when_another_active_request_exists_same_chat() -> None:
    """If the user already has an active session (any mode), the old ML
    timeout notification is suppressed — the user is clearly mid-flow
    and the noise would distract from their current interaction.
    """

    checker, requests, outbox = await _run_checker()

    expired_ml = _make_request(
        req_id="ml-expired",
        chat_id=2002,
        mode=RequestMode.ML,
        status=RequestStatus.PROCESSING,
        timeout_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    # A fresher, still-active session on the same chat (could be ML or
    # MANUAL — the rule is mode-agnostic on the sibling side).
    active_sibling = _make_request(
        req_id="manual-active-sibling",
        chat_id=2002,
        mode=RequestMode.MANUAL,
        status=RequestStatus.PRICING,
        timeout_at=datetime.now(UTC) + timedelta(minutes=5),
        created_at=datetime.now(UTC),
    )
    await requests.add(expired_ml)
    await requests.add(active_sibling)

    await checker._check_timeouts()

    # Expired ML request still got cleaned up
    assert (await requests.get(expired_ml.id)).status is RequestStatus.FAILED
    # Sibling untouched
    assert (await requests.get(active_sibling.id)).status is RequestStatus.PRICING
    # No user-facing notification — sibling-active rule suppressed it.
    assert await outbox.get_unpublished(limit=10) == []


@pytest.mark.anyio
async def test_ml_timeout_emits_notification_when_no_other_active_request() -> None:
    """Happy-path confirmation: a lone ML timeout does notify the user
    and the payload carries ``request_created_at`` so the bot can show
    which of the user's historical requests timed out.
    """

    checker, requests, outbox = await _run_checker()
    created = datetime.now(UTC) - timedelta(minutes=30)
    expired = _make_request(
        req_id="ml-lone",
        chat_id=3003,
        mode=RequestMode.ML,
        status=RequestStatus.PROCESSING,
        timeout_at=datetime.now(UTC) - timedelta(minutes=1),
        created_at=created,
    )
    await requests.add(expired)

    await checker._check_timeouts()

    events = await outbox.get_unpublished(limit=10)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["type"] == "request_timeout"
    assert payload["chat_id"] == 3003
    assert payload["request_id"] == "ml-lone"
    # ISO-8601 with tzinfo — the bot formats this to local HH:MM.
    assert payload["request_created_at"].startswith(created.replace(microsecond=0).isoformat()[:16])


@pytest.mark.anyio
async def test_other_chat_active_session_does_not_suppress_notification() -> None:
    """The active-sibling rule is scoped per-chat. An unrelated user's
    live session must not silence a legitimate ML timeout for a
    different chat — that would be a cross-user information leak of
    behaviour.
    """

    checker, requests, outbox = await _run_checker()

    expired_ml = _make_request(
        req_id="ml-expired-user-A",
        chat_id=7000,
        mode=RequestMode.ML,
        status=RequestStatus.PROCESSING,
        timeout_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    unrelated_active = _make_request(
        req_id="unrelated-user-B",
        chat_id=9999,  # different chat — must NOT count as sibling.
        mode=RequestMode.MANUAL,
        status=RequestStatus.PRICING,
        timeout_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    await requests.add(expired_ml)
    await requests.add(unrelated_active)

    await checker._check_timeouts()

    events = await outbox.get_unpublished(limit=10)
    assert len(events) == 1
    assert events[0].payload["request_id"] == "ml-expired-user-A"
