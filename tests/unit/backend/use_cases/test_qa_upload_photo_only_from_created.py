"""QA: photo uploads must be gated to the CREATED state.

Flow invariant: the bot calls ``POST /v1/requests/{id}/photo`` exactly
once, right after the user sends a photo to a ML-mode request. The
request is then moved ``CREATED -> QUEUED`` and an ``inference_request``
event is written to the outbox. If the use case accepted uploads from
any other state it would either

* re-enqueue a ML request that's already being processed (double-run of
  the model, doubled Kafka traffic, stale ``original_image_key``), or
* poison a manual-mode request by silently replacing its PRICING state
  with QUEUED and orphaning every existing DetectedDamage,
* or worst — resurrect a DONE/FAILED request past its lifecycle.

So "only from CREATED" is the production invariant, and these tests
parametrize every other state to prove the gate does reject them.
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
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus
from auto_repair_estimator.backend.use_cases.repair_requests import (
    UploadPhotoInput,
    UploadPhotoUseCase,
)

# Every non-CREATED state — photo upload must reject each one of them.
_NON_CREATED_STATES = [
    RequestStatus.QUEUED,
    RequestStatus.PROCESSING,
    RequestStatus.PRICING,
    RequestStatus.DONE,
    RequestStatus.FAILED,
]


def _make_request_in_status(status: RequestStatus) -> RepairRequest:
    now = datetime.now(UTC)
    return RepairRequest(
        id="req-upload",
        chat_id=1,
        user_id=2,
        mode=RequestMode.ML,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + timedelta(minutes=5),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("status", _NON_CREATED_STATES, ids=lambda s: s.value)
async def test_upload_photo_is_rejected_from_non_created_states(status: RequestStatus) -> None:
    """Any attempt to upload a photo from a non-CREATED state must raise
    ``ValueError`` BEFORE writing to either the request repo or the outbox."""

    repo = InMemoryRepairRequestRepository()
    outbox = InMemoryOutboxRepository()
    await repo.add(_make_request_in_status(status))
    use_case = UploadPhotoUseCase(
        repository=repo,
        state_machine=RequestStateMachine(),
        outbox_repository=outbox,
        inference_requests_topic="inference_requests",
    )

    with pytest.raises(ValueError, match="CREATED"):
        await use_case.execute(UploadPhotoInput(request_id="req-upload", image_key="k"))

    # The repository must be unchanged.
    stored = await repo.get("req-upload")
    assert stored is not None
    assert stored.status is status, (
        f"UploadPhotoUseCase mutated a {status.value!r} request despite rejecting "
        "it. That would leave the system in a partially-applied state."
    )
    # No outbox event should have been written — otherwise a retried upload
    # would flood Kafka with inference_requests for stale photos.
    unpublished = await outbox.get_unpublished(10)
    assert unpublished == [], (
        "UploadPhotoUseCase wrote to the outbox even though the state check "
        "rejected the call. This would publish an inference_request for a "
        "request that was never transitioned to QUEUED."
    )


@pytest.mark.anyio
async def test_upload_photo_from_manual_mode_is_rejected_before_state_check() -> None:
    """Manual-mode requests start in PRICING and have no photo lifecycle.
    The use case rejects them with a specific 'manual' error — this means
    the user in manual mode won't accidentally trigger the ML pipeline if
    they send a photo attachment too."""

    repo = InMemoryRepairRequestRepository()
    now = datetime.now(UTC)
    manual_req = RepairRequest(
        id="req-manual",
        chat_id=1,
        user_id=2,
        mode=RequestMode.MANUAL,
        # Make the status CREATED to ensure the mode check fires first and
        # returns a MANUAL-specific message, not a state-specific one.
        status=RequestStatus.CREATED,
        created_at=now,
        updated_at=now,
        timeout_at=now + timedelta(minutes=5),
    )
    await repo.add(manual_req)
    use_case = UploadPhotoUseCase(
        repository=repo,
        state_machine=RequestStateMachine(),
        outbox_repository=InMemoryOutboxRepository(),
        inference_requests_topic="inference_requests",
    )

    with pytest.raises(ValueError, match="manual"):
        await use_case.execute(UploadPhotoInput(request_id="req-manual", image_key="k"))
