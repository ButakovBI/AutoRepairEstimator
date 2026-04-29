"""Adversarial tests for ``OutboxFlusher._flush_batch``.

The flusher is the bridge between the transactional outbox (DB-backed) and
Kafka. Two broken behaviours would silently destroy at-least-once guarantees:

1. If a single Kafka publish in the batch fails, **other** successfully
   published events must still be marked as published — otherwise retry
   loops would re-publish them and downstream consumers would see duplicates
   on every subsequent batch.
2. Conversely, events that fail to publish must **not** be marked as
   published, otherwise they'd be lost forever.

We test ``_flush_batch`` directly rather than the ``run`` loop — the loop is
a trivial wrapper around the batch function, and making it an explicit unit
keeps the test deterministic and fast.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_outbox_repository import (
    InMemoryOutboxRepository,
)
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.workers.outbox_flusher import OutboxFlusher


class _StubProducer:
    """Test double for ``KafkaProducer``.

    ``fail_topics`` — any event whose topic is in this set will raise on
    ``send`` so we can simulate partial-batch failures without threading a
    ``MagicMock`` all the way through async code.
    """

    def __init__(self, fail_topics: set[str] | None = None) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self._fail_topics = fail_topics or set()

    async def send(self, topic: str, payload: dict[str, Any]) -> None:
        if topic in self._fail_topics:
            raise RuntimeError(f"simulated broker failure on topic={topic}")
        self.sent.append((topic, payload))


def _event(topic: str, offset_seconds: int = 0, event_id: str | None = None) -> OutboxEvent:
    return OutboxEvent(
        id=event_id or f"evt-{topic}-{offset_seconds}",
        aggregate_id="req-1",
        topic=topic,
        payload={"marker": topic},
        # Distinct timestamps so we can rely on ordering even when events are
        # added in the same test tick.
        created_at=datetime.now(UTC) + timedelta(seconds=offset_seconds),
    )


@pytest.mark.anyio
async def test_empty_outbox_is_a_no_op() -> None:
    outbox = InMemoryOutboxRepository()
    producer = _StubProducer()
    flusher = OutboxFlusher(outbox, producer, batch_size=10)

    await flusher._flush_batch()

    assert producer.sent == []


@pytest.mark.anyio
async def test_flusher_publishes_events_and_marks_them_published() -> None:
    outbox = InMemoryOutboxRepository()
    await outbox.add(_event("inference_requests"))
    await outbox.add(_event("notifications", offset_seconds=1))
    producer = _StubProducer()
    flusher = OutboxFlusher(outbox, producer, batch_size=10)

    await flusher._flush_batch()

    assert {t for t, _ in producer.sent} == {"inference_requests", "notifications"}
    assert await outbox.get_unpublished(limit=10) == []


@pytest.mark.anyio
async def test_failing_events_stay_unpublished_while_others_succeed() -> None:
    outbox = InMemoryOutboxRepository()
    ok_event = _event("inference_requests", offset_seconds=0, event_id="evt-ok")
    bad_event = _event("broken_topic", offset_seconds=1, event_id="evt-bad")
    other_ok_event = _event("notifications", offset_seconds=2, event_id="evt-ok-2")
    await outbox.add(ok_event)
    await outbox.add(bad_event)
    await outbox.add(other_ok_event)
    producer = _StubProducer(fail_topics={"broken_topic"})
    flusher = OutboxFlusher(outbox, producer, batch_size=10)

    await flusher._flush_batch()

    sent_topics = [t for t, _ in producer.sent]
    assert "inference_requests" in sent_topics
    assert "notifications" in sent_topics
    assert "broken_topic" not in sent_topics

    unpublished = await outbox.get_unpublished(limit=10)
    assert [e.id for e in unpublished] == ["evt-bad"], (
        "Only the failing event must remain unpublished so the next flush retries it. "
        f"Instead got: {[(e.id, e.topic) for e in unpublished]}"
    )


@pytest.mark.anyio
async def test_batch_size_limits_how_many_events_are_flushed_in_one_tick() -> None:
    outbox = InMemoryOutboxRepository()
    for i in range(5):
        await outbox.add(_event("notifications", offset_seconds=i, event_id=f"evt-{i}"))
    producer = _StubProducer()
    # Batch size intentionally smaller than outbox — exercises the "limit"
    # argument threaded through to ``get_unpublished``.
    flusher = OutboxFlusher(outbox, producer, batch_size=2)

    await flusher._flush_batch()

    assert len(producer.sent) == 2
    remaining = await outbox.get_unpublished(limit=10)
    assert len(remaining) == 3
    # FIFO: the two oldest (evt-0, evt-1) should have been flushed.
    assert {e.id for e in remaining} == {"evt-2", "evt-3", "evt-4"}
