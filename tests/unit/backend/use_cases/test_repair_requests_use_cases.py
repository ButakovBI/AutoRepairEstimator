from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pytest import mark, raises

from auto_repair_estimator.backend.adapters.repositories.in_memory_outbox_repository import InMemoryOutboxRepository
from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus
from auto_repair_estimator.backend.use_cases.repair_requests import (
    ConfirmPricingInput,
    ConfirmPricingUseCase,
    CreateRepairRequestInput,
    CreateRepairRequestUseCase,
    UploadPhotoInput,
    UploadPhotoUseCase,
)


def _make_upload_use_case(
    repository: InMemoryRepairRequestRepository,
    state_machine: RequestStateMachine,
    outbox: InMemoryOutboxRepository | None = None,
) -> UploadPhotoUseCase:
    return UploadPhotoUseCase(
        repository=repository,
        state_machine=state_machine,
        outbox_repository=outbox if outbox is not None else InMemoryOutboxRepository(),
        inference_requests_topic="inference_requests",
    )


@mark.anyio
async def test_create_repair_request_sets_initial_state_and_persists() -> None:
    repository = InMemoryRepairRequestRepository()
    use_case = CreateRepairRequestUseCase(repository=repository)
    inputs = [
        CreateRepairRequestInput(chat_id=1, user_id=2, mode=RequestMode.ML),
        CreateRepairRequestInput(chat_id=3, user_id=None, mode=RequestMode.MANUAL),
    ]

    results = []
    for data in inputs:
        result = await use_case.execute(data)
        results.append(result)

    assert len(repository.items) == 2
    assert results[0].request.mode is RequestMode.ML
    assert results[0].request.status is RequestStatus.CREATED
    assert results[1].request.mode is RequestMode.MANUAL
    assert results[1].request.status is RequestStatus.PRICING


@mark.anyio
async def test_upload_photo_moves_created_ml_request_to_queued_and_sets_image_key() -> None:
    repository = InMemoryRepairRequestRepository()
    state_machine = RequestStateMachine()
    outbox = InMemoryOutboxRepository()
    create_use_case = CreateRepairRequestUseCase(repository=repository)
    upload_use_case = _make_upload_use_case(repository, state_machine, outbox=outbox)

    create_result = await create_use_case.execute(CreateRepairRequestInput(chat_id=1, user_id=2, mode=RequestMode.ML))
    image_key = "raw-images/request-1.jpg"
    upload_result = await upload_use_case.execute(
        UploadPhotoInput(request_id=create_result.request.id, image_key=image_key)
    )

    assert upload_result.request.status is RequestStatus.QUEUED
    assert upload_result.request.original_image_key == image_key


@mark.anyio
async def test_upload_photo_enqueues_inference_requests_outbox_event() -> None:
    # Arrange
    repository = InMemoryRepairRequestRepository()
    outbox = InMemoryOutboxRepository()
    create_use_case = CreateRepairRequestUseCase(repository=repository)
    upload_use_case = _make_upload_use_case(repository, RequestStateMachine(), outbox=outbox)

    create_result = await create_use_case.execute(CreateRepairRequestInput(chat_id=10, user_id=20, mode=RequestMode.ML))
    image_key = "raw-images/req-10.jpg"

    # Act
    await upload_use_case.execute(UploadPhotoInput(request_id=create_result.request.id, image_key=image_key))

    # Assert — one event, correct topic and payload
    events = await outbox.get_unpublished(10)
    assert len(events) == 1
    assert events[0].topic == "inference_requests"
    assert events[0].payload == {"request_id": create_result.request.id, "image_key": image_key}
    assert events[0].aggregate_id == create_result.request.id


@mark.anyio
async def test_upload_photo_rejects_manual_mode() -> None:
    repository = InMemoryRepairRequestRepository()
    state_machine = RequestStateMachine()
    create_use_case = CreateRepairRequestUseCase(repository=repository)
    upload_use_case = _make_upload_use_case(repository, state_machine)

    create_result = await create_use_case.execute(
        CreateRepairRequestInput(chat_id=1, user_id=2, mode=RequestMode.MANUAL)
    )

    with raises(ValueError):
        await upload_use_case.execute(
            UploadPhotoInput(request_id=create_result.request.id, image_key="raw-images/request-manual.jpg")
        )


@mark.anyio
async def test_upload_photo_manual_mode_does_not_emit_outbox_event() -> None:
    # Arrange
    repository = InMemoryRepairRequestRepository()
    outbox = InMemoryOutboxRepository()
    create_use_case = CreateRepairRequestUseCase(repository=repository)
    upload_use_case = _make_upload_use_case(repository, RequestStateMachine(), outbox=outbox)

    create_result = await create_use_case.execute(
        CreateRepairRequestInput(chat_id=1, user_id=2, mode=RequestMode.MANUAL)
    )

    # Act / Assert
    with raises(ValueError):
        await upload_use_case.execute(
            UploadPhotoInput(request_id=create_result.request.id, image_key="raw-images/x.jpg")
        )
    assert await outbox.get_unpublished(10) == []


@mark.anyio
async def test_confirm_pricing_moves_pricing_to_done() -> None:
    repository = InMemoryRepairRequestRepository()
    state_machine = RequestStateMachine()
    use_case = ConfirmPricingUseCase(repository=repository, state_machine=state_machine)

    now = datetime.now(UTC)
    request = RepairRequest(
        id=str(uuid4()),
        chat_id=1,
        user_id=2,
        mode=RequestMode.MANUAL,
        status=RequestStatus.PRICING,
        created_at=now,
        updated_at=now,
        timeout_at=now + timedelta(minutes=5),
    )
    await repository.add(request)

    result = await use_case.execute(ConfirmPricingInput(request_id=request.id))
    assert result.request.status is RequestStatus.DONE
