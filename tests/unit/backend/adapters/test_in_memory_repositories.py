"""Adversarial tests for the in-memory repositories used in dev/tests.

These doubles are production code (``src/.../adapters/repositories/``) that
back the FastAPI app when no Postgres is available. They must behave
identically to the Postgres implementations on the contracts that the
higher layers rely on — otherwise tests would pass locally but fail against
real infrastructure (or vice versa).

Contracts under test (in one file for locality):

* ``InMemoryOutboxRepository`` returns unpublished events **FIFO by
  ``created_at``** and never more than ``limit``. ``mark_published`` is
  idempotent for unknown ids.
* ``InMemoryRepairRequestRepository.get_timed_out_requests`` excludes DONE
  and FAILED even when their ``timeout_at`` is in the past, and includes
  any non-terminal request whose ``timeout_at`` is past.
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
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


# ---------------------------------------------------------------------------
# InMemoryOutboxRepository
# ---------------------------------------------------------------------------


def _event(event_id: str, created_offset_seconds: int) -> OutboxEvent:
    return OutboxEvent(
        id=event_id,
        aggregate_id="req-1",
        topic="notifications",
        payload={"id": event_id},
        created_at=datetime.now(UTC) + timedelta(seconds=created_offset_seconds),
    )


class TestInMemoryOutboxRepository:
    @pytest.mark.anyio
    async def test_get_unpublished_returns_events_fifo_by_created_at(self) -> None:
        repo = InMemoryOutboxRepository()
        # Add out of order so the test wouldn't pass by accident under
        # insertion order.
        await repo.add(_event("evt-late", created_offset_seconds=10))
        await repo.add(_event("evt-early", created_offset_seconds=0))
        await repo.add(_event("evt-mid", created_offset_seconds=5))

        events = await repo.get_unpublished(limit=10)

        assert [e.id for e in events] == ["evt-early", "evt-mid", "evt-late"]

    @pytest.mark.anyio
    async def test_get_unpublished_respects_limit(self) -> None:
        repo = InMemoryOutboxRepository()
        for i in range(5):
            await repo.add(_event(f"evt-{i}", created_offset_seconds=i))

        events = await repo.get_unpublished(limit=2)

        assert len(events) == 2

    @pytest.mark.anyio
    async def test_published_events_are_excluded_from_get_unpublished(self) -> None:
        repo = InMemoryOutboxRepository()
        await repo.add(_event("evt-1", created_offset_seconds=0))
        await repo.add(_event("evt-2", created_offset_seconds=1))

        await repo.mark_published(["evt-1"])

        remaining = await repo.get_unpublished(limit=10)
        assert [e.id for e in remaining] == ["evt-2"]

    @pytest.mark.anyio
    async def test_mark_published_is_idempotent_for_unknown_ids(self) -> None:
        # Duplicate deliveries from the flusher (e.g. after a Kafka partial
        # acknowledge) must not raise — the repo must silently skip unknown
        # ids. Tested because the source uses a ``dict.get`` and any future
        # refactor that uses ``dict[key]`` would break this contract.
        repo = InMemoryOutboxRepository()
        await repo.add(_event("evt-1", created_offset_seconds=0))

        # Does not raise.
        await repo.mark_published(["evt-1", "evt-does-not-exist"])

        assert await repo.get_unpublished(limit=10) == []


# ---------------------------------------------------------------------------
# InMemoryRepairRequestRepository.get_timed_out_requests
# ---------------------------------------------------------------------------


def _req(*, request_id: str, status: RequestStatus, timeout_offset_seconds: int) -> RepairRequest:
    now = datetime.now(UTC)
    return RepairRequest(
        id=request_id,
        chat_id=1,
        user_id=2,
        mode=RequestMode.ML,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + timedelta(seconds=timeout_offset_seconds),
    )


class TestGetTimedOutRequests:
    @pytest.mark.anyio
    async def test_non_terminal_requests_with_expired_timeout_are_returned(self) -> None:
        repo = InMemoryRepairRequestRepository()
        expired_processing = _req(
            request_id="r-processing",
            status=RequestStatus.PROCESSING,
            timeout_offset_seconds=-60,
        )
        await repo.add(expired_processing)

        result = await repo.get_timed_out_requests()

        assert [r.id for r in result] == ["r-processing"]

    @pytest.mark.anyio
    async def test_done_requests_are_excluded_even_if_timeout_has_passed(self) -> None:
        repo = InMemoryRepairRequestRepository()
        done_expired = _req(
            request_id="r-done",
            status=RequestStatus.DONE,
            timeout_offset_seconds=-300,
        )
        await repo.add(done_expired)

        result = await repo.get_timed_out_requests()

        assert result == []

    @pytest.mark.anyio
    async def test_failed_requests_are_excluded_even_if_timeout_has_passed(self) -> None:
        repo = InMemoryRepairRequestRepository()
        failed_expired = _req(
            request_id="r-failed",
            status=RequestStatus.FAILED,
            timeout_offset_seconds=-300,
        )
        await repo.add(failed_expired)

        result = await repo.get_timed_out_requests()

        assert result == []

    @pytest.mark.anyio
    async def test_fresh_non_terminal_requests_are_excluded(self) -> None:
        repo = InMemoryRepairRequestRepository()
        fresh = _req(
            request_id="r-fresh",
            status=RequestStatus.QUEUED,
            # 60 s in the future — nowhere near timing out.
            timeout_offset_seconds=60,
        )
        await repo.add(fresh)

        result = await repo.get_timed_out_requests()

        assert result == []

    @pytest.mark.anyio
    async def test_expired_queued_processing_created_all_returned(self) -> None:
        """Every non-terminal state with an expired timeout must be returned."""
        repo = InMemoryRepairRequestRepository()
        statuses = [
            RequestStatus.CREATED,
            RequestStatus.QUEUED,
            RequestStatus.PROCESSING,
            RequestStatus.PRICING,
        ]
        for s in statuses:
            await repo.add(
                _req(
                    request_id=f"r-{s.value}",
                    status=s,
                    timeout_offset_seconds=-1,
                )
            )

        result = await repo.get_timed_out_requests()

        assert {r.status for r in result} == set(statuses)
