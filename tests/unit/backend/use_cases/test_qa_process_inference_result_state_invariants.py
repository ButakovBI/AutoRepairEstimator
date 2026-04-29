"""QA: ``ProcessInferenceResultUseCase`` state-machine invariants.

The plan's state machine is explicit (``## State Machine`` section):

* ML flow: ``CREATED -> QUEUED -> PROCESSING -> PRICING -> DONE``
* MANUAL flow: ``CREATED -> PRICING -> DONE`` (Kafka is NOT involved)

Two invariants follow that the current implementation violates:

1. **No ``CREATED -> PRICING`` jump via inference results.** An
   ``inference_results`` message referring to a request still in
   ``CREATED`` is a programming error (the upload-photo step never ran)
   or a replayed stale message. Either way, the use case should ignore
   it, not push the request straight into PRICING through the
   ``CREATED -> PRICING`` edge that the state machine only allows for
   the MANUAL flow.

2. **No inference processing for MANUAL requests.** A MANUAL-mode
   request goes straight from CREATED to PRICING without touching
   Kafka. If the use case nevertheless receives an ``inference_results``
   message for it (cross-mode message, replay, bug), it must be a no-op
   — otherwise it could overwrite manual damages with ML ones.
"""

from __future__ import annotations

import datetime as _dt
from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.detected_part import DetectedPart
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus
from auto_repair_estimator.backend.use_cases.process_inference_result import (
    InferenceDamageData,
    InferencePartData,
    ProcessInferenceResultInput,
    ProcessInferenceResultUseCase,
)


class _FakeRequestRepo:
    def __init__(self) -> None:
        self._items: dict[str, RepairRequest] = {}

    async def add(self, r: RepairRequest) -> None:
        self._items[r.id] = r

    async def get(self, rid: str) -> RepairRequest | None:
        return self._items.get(rid)

    async def update(self, r: RepairRequest) -> None:
        self._items[r.id] = r


class _FakePartRepo:
    def __init__(self) -> None:
        self._items: list[DetectedPart] = []

    async def add(self, p: DetectedPart) -> None:
        self._items.append(p)

    async def get_by_request_id(self, rid: str) -> list[DetectedPart]:
        return [p for p in self._items if p.request_id == rid]


class _FakeDamageRepo:
    def __init__(self) -> None:
        self._items: list[DetectedDamage] = []

    async def add(self, d: DetectedDamage) -> None:
        self._items.append(d)

    async def get_by_request_id(self, rid: str) -> list[DetectedDamage]:
        return [d for d in self._items if d.request_id == rid]

    async def get(self, did: str) -> DetectedDamage | None:
        return next((d for d in self._items if d.id == did), None)

    async def update(self, d: DetectedDamage) -> None:
        for i, item in enumerate(self._items):
            if item.id == d.id:
                self._items[i] = d
                return

    async def soft_delete(self, did: str) -> None:
        for i, item in enumerate(self._items):
            if item.id == did:
                self._items[i] = DetectedDamage(
                    id=item.id,
                    request_id=item.request_id,
                    damage_type=item.damage_type,
                    part_type=item.part_type,
                    source=item.source,
                    is_deleted=True,
                )
                return


class _FakeOutboxRepo:
    def __init__(self) -> None:
        self._items: list[OutboxEvent] = []

    async def add(self, e: OutboxEvent) -> None:
        self._items.append(e)

    async def get_unpublished(self, limit: int) -> list[OutboxEvent]:
        return [e for e in self._items if e.published_at is None][:limit]

    async def mark_published(self, ids: list[str]) -> None:
        return None


def _make_request(req_id: str, status: RequestStatus, mode: RequestMode = RequestMode.ML) -> RepairRequest:
    now = _dt.datetime.now(_dt.UTC)
    return RepairRequest(
        id=req_id,
        chat_id=11,
        user_id=22,
        mode=mode,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )


def _make_use_case(
    req_repo: _FakeRequestRepo,
    part_repo: _FakePartRepo,
    dmg_repo: _FakeDamageRepo,
    outbox_repo: _FakeOutboxRepo,
) -> ProcessInferenceResultUseCase:
    return ProcessInferenceResultUseCase(
        request_repository=req_repo,
        part_repository=part_repo,
        damage_repository=dmg_repo,
        outbox_repository=outbox_repo,
        state_machine=RequestStateMachine(),
        notifications_topic="notifications",
    )


# ---------------------------------------------------------------------------
# Invariant 1: no CREATED -> PRICING jump via inference results
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_inference_result_for_created_request_does_not_skip_to_pricing() -> None:
    """The ML flow requires CREATED -> QUEUED (via UploadPhotoUseCase) before
    any inference_results message can touch the request. Accepting an
    inference result in CREATED effectively skips the QUEUED step and the
    associated outbox inference_request — meaning the ML worker's work
    is attributed to a request that was never actually submitted to Kafka.
    """

    # Arrange
    req_repo = _FakeRequestRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id, RequestStatus.CREATED, RequestMode.ML))
    part_repo = _FakePartRepo()
    dmg_repo = _FakeDamageRepo()
    outbox_repo = _FakeOutboxRepo()
    use_case = _make_use_case(req_repo, part_repo, dmg_repo, outbox_repo)

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[InferencePartData("hood", 0.9, [0.5, 0.5, 0.2, 0.2], "crops/x.jpg")],
            damages=[InferenceDamageData("scratch", "hood", 0.8, None)],
            composited_image_key="composites/x.jpg",
            error_message=None,
        )
    )

    # Assert — request must not have jumped straight to PRICING.
    stored = await req_repo.get(req_id)
    assert stored is not None
    assert stored.status is not RequestStatus.PRICING, (
        "ProcessInferenceResultUseCase accepted an inference result for a request "
        "still in CREATED status and transitioned it directly to PRICING. "
        "This bypasses the QUEUED and PROCESSING states (and thus the outbox "
        "inference_requests event), corrupting the state machine invariants."
    )


# ---------------------------------------------------------------------------
# Invariant 2: MANUAL requests must not be processed as ML results
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_inference_result_for_manual_mode_request_is_ignored() -> None:
    """A MANUAL-mode request goes CREATED -> PRICING without Kafka. If an
    inference_results message arrives for it (stale message, cross-mode
    bug, or a broken producer), the use case must not overwrite manually
    entered damages with ML-derived ones."""

    # Arrange — manual request with one user-entered damage already.
    req_repo = _FakeRequestRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id, RequestStatus.PRICING, RequestMode.MANUAL))
    part_repo = _FakePartRepo()
    dmg_repo = _FakeDamageRepo()
    outbox_repo = _FakeOutboxRepo()
    use_case = _make_use_case(req_repo, part_repo, dmg_repo, outbox_repo)

    # Act — inference_results payload targets the manual request by mistake.
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[InferencePartData("hood", 0.9, [0.5, 0.5, 0.2, 0.2], "crops/x.jpg")],
            damages=[InferenceDamageData("scratch", "hood", 0.8, None)],
            composited_image_key="composites/x.jpg",
            error_message=None,
        )
    )

    # Assert — no ML parts or damages should have been persisted, and no
    # notification should have been enqueued for a manual flow.
    assert await part_repo.get_by_request_id(req_id) == [], (
        "ProcessInferenceResultUseCase persisted ML-derived parts for a MANUAL-mode "
        "request. Manual sessions should be opaque to the inference pipeline."
    )
    assert await dmg_repo.get_by_request_id(req_id) == [], (
        "ProcessInferenceResultUseCase persisted ML-derived damages for a MANUAL-mode "
        "request; user-entered damages could be overwritten by ML ones on redelivery."
    )
    assert await outbox_repo.get_unpublished(10) == [], (
        "ProcessInferenceResultUseCase enqueued a notification for a MANUAL-mode "
        "request, creating a spurious 'inference_complete'/'inference_failed' "
        "message that the user did not expect."
    )
