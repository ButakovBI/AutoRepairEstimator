"""Adversarial tests for ``HeartbeatChecker._check_timeouts``.

Behavioral contracts we care about:

1. When there are no timed-out requests, neither the request repo nor the
   outbox must be written to — otherwise we'd emit spurious timeout events.
2. A timed-out request moves from its current non-terminal status to
   ``FAILED`` AND an ``outbox_event`` of type ``request_timeout`` is
   enqueued. Both must happen (the UI depends on the event; the DB row is
   the source of truth for downstream queries).
3. If the state machine refuses a transition (e.g. the request moved to
   ``DONE`` concurrently after ``get_timed_out_requests`` but before the
   checker touched it), the checker must **not** crash the loop nor enqueue
   a timeout event — the checker is a background worker and any unhandled
   exception would kill the entire backend task supervision tree.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_outbox_repository import (
    InMemoryOutboxRepository,
)
from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus
from auto_repair_estimator.backend.workers.heartbeat_checker import HeartbeatChecker


def _make_request(*, status: RequestStatus, timeout_at: datetime) -> RepairRequest:
    now = datetime.now(UTC)
    return RepairRequest(
        id=f"req-{status.value}",
        chat_id=123,
        user_id=456,
        mode=RequestMode.ML,
        status=status,
        created_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=5),
        timeout_at=timeout_at,
    )


async def _build_checker() -> tuple[
    HeartbeatChecker, InMemoryRepairRequestRepository, InMemoryOutboxRepository
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
async def test_no_timed_out_requests_produces_no_outbox_events() -> None:
    checker, requests, outbox = await _build_checker()
    # Future-dated timeout: not expired yet.
    fresh = _make_request(
        status=RequestStatus.PROCESSING,
        timeout_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    await requests.add(fresh)

    await checker._check_timeouts()

    assert await outbox.get_unpublished(limit=10) == []
    assert (await requests.get(fresh.id)).status is RequestStatus.PROCESSING


@pytest.mark.anyio
async def test_timed_out_request_transitions_to_failed_and_emits_notification() -> None:
    checker, requests, outbox = await _build_checker()
    expired = _make_request(
        status=RequestStatus.PROCESSING,
        timeout_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    await requests.add(expired)

    await checker._check_timeouts()

    stored = await requests.get(expired.id)
    assert stored is not None
    assert stored.status is RequestStatus.FAILED

    events = await outbox.get_unpublished(limit=10)
    assert len(events) == 1
    assert events[0].topic == "notifications"
    assert events[0].payload == {
        "chat_id": expired.chat_id,
        "request_id": expired.id,
        "type": "request_timeout",
    }


@pytest.mark.anyio
async def test_terminal_requests_are_not_returned_by_timeout_scan() -> None:
    # Sanity check that even if a DONE request has an expired ``timeout_at``,
    # the checker does not try to transition it again. The filtering lives
    # in ``get_timed_out_requests`` — we verify the checker honours it.
    checker, requests, outbox = await _build_checker()
    done = _make_request(
        status=RequestStatus.DONE,
        timeout_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await requests.add(done)

    await checker._check_timeouts()

    assert (await requests.get(done.id)).status is RequestStatus.DONE
    assert await outbox.get_unpublished(limit=10) == []


@pytest.mark.anyio
async def test_concurrent_state_change_does_not_crash_the_loop() -> None:
    """If a request becomes terminal between the scan and the transition,
    the state machine will raise — the checker must swallow that and
    continue processing the rest of the batch.
    """

    checker, requests, outbox = await _build_checker()

    # Use a repo subclass that returns a list containing a DONE request
    # (simulating a race where get_timed_out_requests is stale) alongside a
    # genuinely timed-out one. The DONE transition must raise; the other
    # must still be processed.
    class _StaleScanRepo(InMemoryRepairRequestRepository):
        def __init__(self, stale: list[RepairRequest]) -> None:
            super().__init__()
            self._stale = stale

        async def get_timed_out_requests(self) -> list[RepairRequest]:
            return list(self._stale)

    done = _make_request(
        status=RequestStatus.DONE,
        timeout_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    processing = _make_request(
        status=RequestStatus.PROCESSING,
        timeout_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    # Distinct ids so the repo can hold both.
    processing = RepairRequest(
        id="req-processing-race",
        chat_id=processing.chat_id,
        user_id=processing.user_id,
        mode=processing.mode,
        status=processing.status,
        created_at=processing.created_at,
        updated_at=processing.updated_at,
        timeout_at=processing.timeout_at,
    )

    stale_repo = _StaleScanRepo([done, processing])
    await stale_repo.add(done)
    await stale_repo.add(processing)

    racing_checker = HeartbeatChecker(
        request_repository=stale_repo,
        outbox_repository=outbox,
        state_machine=RequestStateMachine(),
        notifications_topic="notifications",
    )

    # Must not raise.
    await racing_checker._check_timeouts()

    # DONE must remain DONE — transition was invalid and swallowed.
    assert (await stale_repo.get(done.id)).status is RequestStatus.DONE
    # The legitimate expired PROCESSING request still got marked FAILED.
    assert (await stale_repo.get(processing.id)).status is RequestStatus.FAILED

    # Exactly one timeout notification — for the legit request.
    events = await outbox.get_unpublished(limit=10)
    assert len(events) == 1
    assert events[0].payload["request_id"] == processing.id


# Silence asyncio warnings because these tests are event-loop based. AnyIO
# wires the loop automatically via ``pytest.mark.anyio``.
def _touch_unused_symbols() -> None:  # pragma: no cover - import-time guard only
    _ = (Any,)
