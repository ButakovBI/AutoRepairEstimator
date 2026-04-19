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


def _make_request(req_id: str, status: RequestStatus = RequestStatus.PROCESSING) -> RepairRequest:
    import datetime as _dt

    now = _dt.datetime.now(_dt.UTC)
    return RepairRequest(
        id=req_id,
        chat_id=123,
        user_id=456,
        mode=RequestMode.ML,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )


def _make_processing_request(req_id: str) -> RepairRequest:
    return _make_request(req_id, RequestStatus.PROCESSING)


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
    already_priced = _make_request(req_id, RequestStatus.PRICING)
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
async def test_queued_request_is_transitioned_through_processing_to_pricing() -> None:
    """Backend must accept inference results for QUEUED requests by first moving them to PROCESSING.

    This matches the plan's state diagram (CREATED -> QUEUED -> PROCESSING -> PRICING)
    without requiring the stateless ML worker to write to the DB.
    """
    # Arrange
    req_repo = _FakeRequestRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id, RequestStatus.QUEUED))
    use_case = _make_use_case(req_repo, _FakePartRepo(), _FakeDamageRepo(), _FakeOutboxRepo())

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

    # Assert
    updated = await req_repo.get(req_id)
    assert updated is not None
    assert updated.status is RequestStatus.PRICING


@pytest.mark.anyio
async def test_done_request_is_not_updated_on_late_inference_result() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    outbox_repo = _FakeOutboxRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id, RequestStatus.DONE))
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

    # Assert — terminal DONE is not touched and no extra notification is enqueued
    updated = await req_repo.get(req_id)
    assert updated is not None
    assert updated.status is RequestStatus.DONE
    assert await outbox_repo.get_unpublished(10) == []


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


# ---------------------------------------------------------------------------
# Adversarial idempotency: redelivery of an inference result for a request
# that is already in PROCESSING state with persisted parts/damages.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_redelivery_to_processing_request_does_not_duplicate_parts_or_damages() -> None:
    """Scenario: first delivery persisted parts and damages but crashed
    before transitioning to PRICING (DB commit succeeded, Kafka commit of
    the consumer offset failed). Kafka then redelivers. The use case must
    finish the transition without re-inserting rows, otherwise each retry
    would double (then quadruple) the dataset.
    """

    # Arrange — simulate the "first delivery partially completed" world:
    # request is PROCESSING and part_repo already contains one part.
    req_repo = _FakeRequestRepo()
    part_repo = _FakePartRepo()
    dmg_repo = _FakeDamageRepo()
    outbox_repo = _FakeOutboxRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    # Pre-populate parts — this is how we detect "already processed".
    from auto_repair_estimator.backend.domain.value_objects.request_enums import PartType

    await part_repo.add(
        DetectedPart(
            id=str(uuid4()),
            request_id=req_id,
            part_type=PartType.HOOD,
            confidence=0.9,
            x=0.0,
            y=0.0,
            width=1.0,
            height=1.0,
            crop_image_key="crops/x.jpg",
        )
    )

    use_case = _make_use_case(req_repo, part_repo, dmg_repo, outbox_repo)

    # Act — redelivered message with same payload.
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
    stored = await req_repo.get(req_id)
    assert stored is not None
    assert stored.status is RequestStatus.PRICING, (
        "Redelivery must still finish the transition to PRICING so the user is not stuck."
    )

    parts_after = await part_repo.get_by_request_id(req_id)
    assert len(parts_after) == 1, (
        "Redelivery must not duplicate persisted parts; got "
        f"{len(parts_after)} rows, expected 1 (the one seeded before redelivery)."
    )
    damages_after = await dmg_repo.get_by_request_id(req_id)
    assert len(damages_after) == 0, (
        "Redelivery must not insert new damages either — the first delivery "
        "is responsible for those."
    )

    # The redelivery still emits exactly one notification so the bot knows
    # to proceed. Any more would spam the user with duplicate messages.
    events = await outbox_repo.get_unpublished(10)
    assert len(events) == 1
    assert events[0].payload["type"] == "inference_complete"


@pytest.mark.anyio
async def test_unknown_part_type_in_payload_is_skipped_not_crashed() -> None:
    """An ML model upgrade (or message corruption) may emit a part_type
    value that isn't in the user-approved ``PartType`` enum. The use case
    must skip that detection with a warning rather than letting the
    PartType(...) constructor bubble a ValueError to the Kafka consumer
    and kill the whole consumer loop.
    """

    req_repo = _FakeRequestRepo()
    part_repo = _FakePartRepo()
    dmg_repo = _FakeDamageRepo()
    outbox_repo = _FakeOutboxRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_processing_request(req_id))
    use_case = _make_use_case(req_repo, part_repo, dmg_repo, outbox_repo)

    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[
                InferencePartData("hood", 0.9, [0.5, 0.5, 0.2, 0.2], "crops/a.jpg"),
                # Unknown part class in a newer YOLO model.
                InferencePartData("spoiler", 0.9, [0.5, 0.5, 0.2, 0.2], "crops/b.jpg"),
            ],
            damages=[
                InferenceDamageData("scratch", "hood", 0.8, None),
                # Damage linked to unknown part — also skipped.
                InferenceDamageData("scratch", "spoiler", 0.8, None),
            ],
            composited_image_key="composites/x.jpg",
            error_message=None,
        )
    )

    parts = await part_repo.get_by_request_id(req_id)
    damages = await dmg_repo.get_by_request_id(req_id)
    assert len(parts) == 1, f"Expected the unknown 'spoiler' part to be skipped; got {parts}."
    assert len(damages) == 1, f"Expected the unknown-part damage to be skipped; got {damages}."

    # Request still progresses to PRICING so the user isn't stuck forever.
    stored = await req_repo.get(req_id)
    assert stored is not None
    assert stored.status is RequestStatus.PRICING
