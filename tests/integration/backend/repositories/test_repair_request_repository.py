"""Behavioral integration tests for PostgresRepairRequestRepository.

Each test verifies a single observable behaviour (what the repository does
from the caller's perspective), not how it is implemented internally.
All tests follow the AAA (Arrange / Act / Assert) convention.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
import pytest

from auto_repair_estimator.backend.adapters.repositories.postgres_repair_request_repository import (
    PostgresRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


def _make_request(
    *,
    mode: RequestMode = RequestMode.MANUAL,
    status: RequestStatus = RequestStatus.CREATED,
    chat_id: int = 111,
    timeout_delta: timedelta = timedelta(minutes=5),
) -> RepairRequest:
    now = datetime.now(UTC)
    return RepairRequest(
        id=str(uuid4()),
        chat_id=chat_id,
        user_id=None,
        mode=mode,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + timeout_delta,
    )


@pytest.mark.anyio
async def test_added_request_can_be_retrieved_by_id(db_pool: asyncpg.Pool) -> None:
    # Arrange
    repo = PostgresRepairRequestRepository(db_pool)
    request = _make_request(mode=RequestMode.ML, status=RequestStatus.CREATED)

    # Act
    await repo.add(request)
    fetched = await repo.get(request.id)

    # Assert
    assert fetched is not None
    assert fetched.id == request.id
    assert fetched.mode is RequestMode.ML
    assert fetched.status is RequestStatus.CREATED
    assert fetched.chat_id == 111


@pytest.mark.anyio
async def test_get_returns_none_for_unknown_id(db_pool: asyncpg.Pool) -> None:
    # Arrange
    repo = PostgresRepairRequestRepository(db_pool)

    # Act
    result = await repo.get(str(uuid4()))

    # Assert
    assert result is None


@pytest.mark.anyio
async def test_status_change_is_persisted_after_update(db_pool: asyncpg.Pool) -> None:
    # Arrange
    repo = PostgresRepairRequestRepository(db_pool)
    request = _make_request(status=RequestStatus.CREATED)
    await repo.add(request)

    # Act — mutate then persist
    request.status = RequestStatus.QUEUED
    await repo.update(request)
    refreshed = await repo.get(request.id)

    # Assert
    assert refreshed is not None
    assert refreshed.status is RequestStatus.QUEUED


@pytest.mark.anyio
async def test_image_key_is_persisted_after_update(db_pool: asyncpg.Pool) -> None:
    # Arrange
    repo = PostgresRepairRequestRepository(db_pool)
    request = _make_request(mode=RequestMode.ML)
    await repo.add(request)

    # Act
    request.original_image_key = "raw-images/car.jpg"
    await repo.update(request)
    refreshed = await repo.get(request.id)

    # Assert
    assert refreshed is not None
    assert refreshed.original_image_key == "raw-images/car.jpg"


@pytest.mark.anyio
async def test_timed_out_request_is_returned_by_watchdog_query(db_pool: asyncpg.Pool) -> None:
    # Arrange — one request already past its deadline, one still active
    repo = PostgresRepairRequestRepository(db_pool)

    expired = _make_request(
        mode=RequestMode.ML,
        status=RequestStatus.PROCESSING,
        timeout_delta=timedelta(minutes=-5),  # timeout in the past
    )
    active = _make_request(
        mode=RequestMode.ML,
        status=RequestStatus.PROCESSING,
        timeout_delta=timedelta(minutes=+5),  # timeout in the future
    )
    await repo.add(expired)
    await repo.add(active)

    # Act
    timed_out = await repo.get_timed_out_requests()

    # Assert — only the expired request surfaces
    ids = {r.id for r in timed_out}
    assert expired.id in ids
    assert active.id not in ids


@pytest.mark.anyio
async def test_done_requests_are_excluded_from_timeout_check(db_pool: asyncpg.Pool) -> None:
    # Arrange — DONE request whose timeout_at is already in the past
    repo = PostgresRepairRequestRepository(db_pool)
    done_request = _make_request(
        mode=RequestMode.MANUAL,
        status=RequestStatus.DONE,
        timeout_delta=timedelta(minutes=-5),
    )
    await repo.add(done_request)

    # Act
    timed_out = await repo.get_timed_out_requests()

    # Assert — terminal-status requests must never be flagged for timeout
    assert all(r.id != done_request.id for r in timed_out)


@pytest.mark.anyio
async def test_failed_requests_are_excluded_from_timeout_check(db_pool: asyncpg.Pool) -> None:
    # Arrange
    repo = PostgresRepairRequestRepository(db_pool)
    failed = _make_request(status=RequestStatus.FAILED, timeout_delta=timedelta(minutes=-5))
    await repo.add(failed)

    # Act
    timed_out = await repo.get_timed_out_requests()

    # Assert
    assert all(r.id != failed.id for r in timed_out)
