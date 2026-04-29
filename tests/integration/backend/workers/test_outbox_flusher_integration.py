"""End-to-end integration test for ``OutboxFlusher`` against a real Postgres.

Ensures the flusher + Postgres outbox repository cooperate correctly:

* ordering by ``created_at`` (FIFO) is enforced by the SQL query, not only
  the in-memory fake;
* ``mark_published`` actually writes the timestamp so the row disappears
  from ``get_unpublished`` on subsequent polls;
* a publisher failure on one event in the batch leaves that row unpublished
  in the database, so the next flusher tick retries it.

Skipped automatically when no Postgres is available (see
``tests/integration/conftest.py`` skip logic).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from auto_repair_estimator.backend.adapters.repositories.postgres_outbox_repository import (
    PostgresOutboxRepository,
)
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.workers.outbox_flusher import OutboxFlusher


class _StubProducer:
    def __init__(self, fail_topics: set[str] | None = None) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self._fail_topics = fail_topics or set()

    async def send(self, topic: str, payload: dict[str, Any]) -> None:
        if topic in self._fail_topics:
            raise RuntimeError(f"broker down for topic={topic}")
        self.sent.append((topic, payload))


def _uuid(seed: int) -> str:
    # Deterministic UUID literals so test failures are easy to bisect.
    return f"00000000-0000-0000-0000-0000000000{seed:02d}"


async def _insert_request_row(pool: asyncpg.Pool, request_id: str) -> None:
    """Outbox events have a FK-style aggregate_id to a request — keep it
    valid by seeding a minimal request row.
    """
    now = datetime.now(UTC)
    await pool.execute(
        """
        INSERT INTO repair_requests (id, chat_id, user_id, mode, status, created_at, updated_at, timeout_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        request_id,
        1,
        2,
        "ml",
        "queued",
        now,
        now,
        now + timedelta(minutes=5),
    )


@pytest.mark.anyio
async def test_flusher_publishes_postgres_backed_events_in_fifo_order(
    db_pool: asyncpg.Pool,
) -> None:
    # Arrange
    request_id = _uuid(1)
    await _insert_request_row(db_pool, request_id)

    repo = PostgresOutboxRepository(db_pool)
    now = datetime.now(UTC)

    # Insert intentionally out of insertion order relative to created_at.
    await repo.add(
        OutboxEvent(
            id=_uuid(11),
            aggregate_id=request_id,
            topic="notifications",
            payload={"marker": "late"},
            created_at=now + timedelta(seconds=10),
        )
    )
    await repo.add(
        OutboxEvent(
            id=_uuid(12),
            aggregate_id=request_id,
            topic="notifications",
            payload={"marker": "early"},
            created_at=now,
        )
    )

    producer = _StubProducer()
    flusher = OutboxFlusher(repo, producer, batch_size=10, poll_interval_ms=1)

    # Act
    await flusher._flush_batch()

    # Assert — FIFO by created_at (the early row comes first).
    assert [p["marker"] for _, p in producer.sent] == ["early", "late"]

    # Both rows must be marked published.
    remaining = await repo.get_unpublished(limit=10)
    assert remaining == []


@pytest.mark.anyio
async def test_flusher_leaves_failing_rows_unpublished_in_postgres(
    db_pool: asyncpg.Pool,
) -> None:
    # Arrange — three rows; one topic will fail to publish.
    request_id = _uuid(2)
    await _insert_request_row(db_pool, request_id)
    repo = PostgresOutboxRepository(db_pool)
    now = datetime.now(UTC)

    ok_id = _uuid(21)
    bad_id = _uuid(22)
    other_ok_id = _uuid(23)

    await repo.add(
        OutboxEvent(id=ok_id, aggregate_id=request_id, topic="notifications", payload={"a": 1}, created_at=now)
    )
    await repo.add(
        OutboxEvent(
            id=bad_id, aggregate_id=request_id, topic="broken", payload={"a": 2}, created_at=now + timedelta(seconds=1)
        )
    )
    await repo.add(
        OutboxEvent(
            id=other_ok_id,
            aggregate_id=request_id,
            topic="notifications",
            payload={"a": 3},
            created_at=now + timedelta(seconds=2),
        )
    )

    producer = _StubProducer(fail_topics={"broken"})
    flusher = OutboxFlusher(repo, producer, batch_size=10, poll_interval_ms=1)

    # Act
    await flusher._flush_batch()

    # Assert — only the failed row remains unpublished in the database.
    remaining = await repo.get_unpublished(limit=10)
    assert [e.id for e in remaining] == [bad_id]

    # And the producer saw the two good events (the bad one raised).
    payloads = [p["a"] for _, p in producer.sent]
    assert sorted(payloads) == [1, 3]


@pytest.mark.anyio
async def test_flusher_retries_failed_row_on_next_tick(db_pool: asyncpg.Pool) -> None:
    """After the broker comes back, the next flush publishes the row."""
    request_id = _uuid(3)
    await _insert_request_row(db_pool, request_id)
    repo = PostgresOutboxRepository(db_pool)
    now = datetime.now(UTC)

    bad_id = _uuid(31)
    await repo.add(
        OutboxEvent(id=bad_id, aggregate_id=request_id, topic="broken", payload={"retry": True}, created_at=now)
    )

    producer = _StubProducer(fail_topics={"broken"})
    flusher = OutboxFlusher(repo, producer, batch_size=10, poll_interval_ms=1)

    await flusher._flush_batch()
    assert (await repo.get_unpublished(limit=10))[0].id == bad_id

    # Broker recovers.
    producer._fail_topics = set()
    await flusher._flush_batch()

    assert await repo.get_unpublished(limit=10) == []
    assert len(producer.sent) == 1


def _touch() -> None:  # pragma: no cover
    # Silence the unused-import warning for asyncio (needed for anyio
    # plugin resolution on some Windows/py3.12 combos).
    _ = asyncio.sleep
