from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pytest import mark, raises

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


@mark.anyio
async def test_create_repair_request_sets_initial_state_and_persists() -> None:
    repository = InMemoryRepairRequestRepository()
    use_case = CreateRepairRequestUseCase(repository=repository)
    inputs = [
        CreateRepairRequestInput(telegram_chat_id=1, telegram_user_id=2, mode=RequestMode.ML),
        CreateRepairRequestInput(telegram_chat_id=3, telegram_user_id=None, mode=RequestMode.MANUAL),
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
    create_use_case = CreateRepairRequestUseCase(repository=repository)
    upload_use_case = UploadPhotoUseCase(repository=repository, state_machine=state_machine)

    create_result = await create_use_case.execute(
        CreateRepairRequestInput(telegram_chat_id=1, telegram_user_id=2, mode=RequestMode.ML)
    )
    image_key = "raw-images/request-1.jpg"
    upload_result = await upload_use_case.execute(
        UploadPhotoInput(request_id=create_result.request.id, image_key=image_key)
    )

    assert upload_result.request.status is RequestStatus.QUEUED
    assert upload_result.request.original_image_key == image_key


@mark.anyio
async def test_upload_photo_rejects_manual_mode() -> None:
    repository = InMemoryRepairRequestRepository()
    state_machine = RequestStateMachine()
    create_use_case = CreateRepairRequestUseCase(repository=repository)
    upload_use_case = UploadPhotoUseCase(repository=repository, state_machine=state_machine)

    create_result = await create_use_case.execute(
        CreateRepairRequestInput(telegram_chat_id=1, telegram_user_id=2, mode=RequestMode.MANUAL)
    )

    with raises(ValueError):
        await upload_use_case.execute(
            UploadPhotoInput(request_id=create_result.request.id, image_key="raw-images/request-manual.jpg")
        )


@mark.anyio
async def test_confirm_pricing_moves_pricing_to_done() -> None:
    repository = InMemoryRepairRequestRepository()
    state_machine = RequestStateMachine()
    use_case = ConfirmPricingUseCase(repository=repository, state_machine=state_machine)

    now = datetime.now(timezone.utc)
    request = RepairRequest(
        id=str(uuid4()),
        telegram_chat_id=1,
        telegram_user_id=2,
        mode=RequestMode.MANUAL,
        status=RequestStatus.PRICING,
        created_at=now,
        updated_at=now,
        timeout_at=now + timedelta(minutes=5),
    )
    await repository.add(request)

    result = await use_case.execute(ConfirmPricingInput(request_id=request.id))
    assert result.request.status is RequestStatus.DONE
    assert result.total_cost == 0.0
    assert result.total_hours == 0.0

