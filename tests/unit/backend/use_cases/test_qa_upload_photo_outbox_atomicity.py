"""QA: UploadPhotoUseCase must be atomic with the outbox insert.

The plan (Phase 2, Week 4) states:

    Update UploadPhotoUseCase: в той же DB-транзакции
    UPDATE request + INSERT outbox_event(topic=inference_requests)

This is the whole point of the Transactional Outbox pattern: the request
transitions from ``CREATED`` to ``QUEUED`` **iff** a matching
``inference_requests`` outbox event is enqueued. Otherwise the request is
orphaned — it sits in ``QUEUED`` forever, never picked up by any ML worker,
because no event was ever published.

The current implementation performs two independent awaits on the
repository layer, with no transaction wrapping. So if the outbox insert
fails after the request update succeeds, the request is permanently stuck.

This test simulates that failure and asserts the atomicity contract.
"""

from __future__ import annotations

import datetime as _dt
from uuid import uuid4

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus
from auto_repair_estimator.backend.use_cases.repair_requests import UploadPhotoInput, UploadPhotoUseCase


class _ExplodingOutboxRepo:
    """Outbox repository whose ``add`` always fails — simulating e.g. a
    network blip to Postgres after the request row was already UPDATEd."""

    def __init__(self) -> None:
        self.added_events: list[OutboxEvent] = []

    async def add(self, event: OutboxEvent) -> None:
        self.added_events.append(event)
        raise RuntimeError("simulated outbox failure")

    async def get_unpublished(self, limit: int) -> list[OutboxEvent]:
        return []

    async def mark_published(self, event_ids: list[str]) -> None:
        return None


def _seed_created_ml_request(repo: InMemoryRepairRequestRepository) -> RepairRequest:
    now = _dt.datetime.now(_dt.UTC)
    request = RepairRequest(
        id=str(uuid4()),
        chat_id=1,
        user_id=2,
        mode=RequestMode.ML,
        status=RequestStatus.CREATED,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )
    # Direct dict access to keep the seeding synchronous.
    repo._items[request.id] = request  # type: ignore[attr-defined]  # noqa: SLF001
    return request


@pytest.mark.anyio
async def test_upload_photo_does_not_leave_request_queued_if_outbox_insert_fails() -> None:
    """The business invariant: a request is QUEUED *only* when an
    ``inference_requests`` event has been enqueued. If the enqueue fails,
    the request must stay in CREATED (or be rolled back to CREATED) so
    a retry can succeed; otherwise the request is lost forever."""

    # Arrange
    repo = InMemoryRepairRequestRepository()
    request = _seed_created_ml_request(repo)
    use_case = UploadPhotoUseCase(
        repository=repo,
        state_machine=RequestStateMachine(),
        outbox_repository=_ExplodingOutboxRepo(),
        inference_requests_topic="inference_requests",
    )

    # Act
    with pytest.raises(RuntimeError, match="simulated outbox failure"):
        await use_case.execute(UploadPhotoInput(request_id=request.id, image_key="raw/1.jpg"))

    # Assert — the request must not have been left in QUEUED with no outbox event.
    after = await repo.get(request.id)
    assert after is not None
    assert after.status is RequestStatus.CREATED, (
        f"UploadPhotoUseCase left the request in status={after.status.value} "
        "after the outbox insert failed. This is a transactional-outbox violation: "
        "no Kafka event will ever be published, so no ML worker will pick up the "
        "request — it is orphaned in QUEUED forever. Wrap the repository update "
        "and the outbox add in a single transaction (see plan, Phase 2 Week 4)."
    )
