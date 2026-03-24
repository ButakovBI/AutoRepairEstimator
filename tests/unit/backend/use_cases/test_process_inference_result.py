"""Unit tests for ProcessInferenceResultUseCase.

Each test verifies exactly one observable behaviour, using in-memory fakes so
the tests stay independent of any database or infrastructure.
AAA structure (Arrange / Act / Assert) is explicit in every test.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


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
        pass


def _make_processing_request(req_id: str) -> RepairRequest:
    return RepairRequest(
        id=req_id,
        chat_id=123,
        user_id=456,
        mode=RequestMode.ML,
        status=RequestStatus.PROCESSING,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        timeout_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        + __import__("datetime").timedelta(minutes=5),
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
# Successful inference
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_successful_inference_transitions_request_to_pricing() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    use_case = _make_use_case(req_repo, _FakePartRepo(), _FakeDamageRepo(), _FakeOutboxRepo())

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[InferencePartData("hood", 0.92, [0.0, 0.0, 1.0, 1.0], "crops/x.jpg")],
            damages=[InferenceDamageData("scratch", "hood", 0.85, None)],
            composited_image_key="composites/x.jpg",
            error_message=None,
        )
    )

    # Assert
    updated = await req_repo.get(req_id)
    assert updated is not None
    assert updated.status is RequestStatus.PRICING


@pytest.mark.anyio
async def test_successful_inference_stores_composited_image_key() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    use_case = _make_use_case(req_repo, _FakePartRepo(), _FakeDamageRepo(), _FakeOutboxRepo())

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[],
            damages=[],
            composited_image_key="composites/car-result.jpg",
            error_message=None,
        )
    )

    # Assert
    updated = await req_repo.get(req_id)
    assert updated is not None
    assert updated.composited_image_key == "composites/car-result.jpg"


@pytest.mark.anyio
async def test_successful_inference_persists_detected_parts_and_damages() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    part_repo = _FakePartRepo()
    dmg_repo = _FakeDamageRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    use_case = _make_use_case(req_repo, part_repo, dmg_repo, _FakeOutboxRepo())

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[InferencePartData("hood", 0.92, [0.5, 0.5, 0.2, 0.2], "crops/x.jpg")],
            damages=[InferenceDamageData("scratch", "hood", 0.85, None)],
            composited_image_key="composites/x.jpg",
            error_message=None,
        )
    )

    # Assert
    assert len(await part_repo.get_by_request_id(req_id)) == 1
    assert len(await dmg_repo.get_by_request_id(req_id)) == 1


@pytest.mark.anyio
async def test_successful_inference_enqueues_inference_complete_notification() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    outbox_repo = _FakeOutboxRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    use_case = _make_use_case(req_repo, _FakePartRepo(), _FakeDamageRepo(), outbox_repo)

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[],
            damages=[],
            composited_image_key="composites/x.jpg",
            error_message=None,
        )
    )

    # Assert — exactly one notification, typed correctly
    events = await outbox_repo.get_unpublished(10)
    assert len(events) == 1
    assert events[0].payload["type"] == "inference_complete"


# ---------------------------------------------------------------------------
# Failed inference
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_failed_inference_transitions_to_pricing_for_manual_fallback() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    use_case = _make_use_case(req_repo, _FakePartRepo(), _FakeDamageRepo(), _FakeOutboxRepo())

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="error",
            parts=[],
            damages=[],
            composited_image_key=None,
            error_message="no_parts_detected",
        )
    )

    # Assert — request is still moved to PRICING so user can input manually
    updated = await req_repo.get(req_id)
    assert updated is not None
    assert updated.status is RequestStatus.PRICING


@pytest.mark.anyio
async def test_failed_inference_enqueues_inference_failed_notification() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    outbox_repo = _FakeOutboxRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    use_case = _make_use_case(req_repo, _FakePartRepo(), _FakeDamageRepo(), outbox_repo)

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="error",
            parts=[],
            damages=[],
            composited_image_key=None,
            error_message="no_parts_detected",
        )
    )

    # Assert
    events = await outbox_repo.get_unpublished(10)
    assert len(events) == 1
    assert events[0].payload["type"] == "inference_failed"


# ---------------------------------------------------------------------------
# Idempotency / edge cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_processing_already_priced_request_is_a_no_op() -> None:
    """Re-delivering an inference result for a PRICING request must be idempotent."""
    # Arrange
    req_repo = _FakeRequestRepo()
    part_repo = _FakePartRepo()
    outbox_repo = _FakeOutboxRepo()
    req_id = str(uuid4())
    already_priced = _make_processing_request(req_id)
    already_priced.status = RequestStatus.PRICING
    await req_repo.add(already_priced)
    use_case = _make_use_case(req_repo, part_repo, _FakeDamageRepo(), outbox_repo)

    # Act
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[InferencePartData("hood", 0.9, [0.0, 0.0, 1.0, 1.0], "crops/x.jpg")],
            damages=[],
            composited_image_key="composites/x.jpg",
            error_message=None,
        )
    )

    # Assert — no side effects applied a second time
    assert len(await part_repo.get_by_request_id(req_id)) == 0
    assert len(await outbox_repo.get_unpublished(10)) == 0


@pytest.mark.anyio
async def test_processing_result_for_unknown_request_produces_no_side_effects() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    outbox_repo = _FakeOutboxRepo()
    use_case = _make_use_case(req_repo, _FakePartRepo(), _FakeDamageRepo(), outbox_repo)

    # Act — no request was seeded
    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=str(uuid4()),
            status="success",
            parts=[],
            damages=[],
            composited_image_key=None,
            error_message=None,
        )
    )

    # Assert — nothing was written
    assert len(await outbox_repo.get_unpublished(10)) == 0
