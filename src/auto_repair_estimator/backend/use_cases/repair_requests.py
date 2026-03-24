from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


@dataclass
class CreateRepairRequestInput:
    chat_id: int
    user_id: int | None
    mode: RequestMode


@dataclass
class CreateRepairRequestResult:
    request: RepairRequest


class CreateRepairRequestUseCase:
    def __init__(self, repository: RepairRequestRepository) -> None:
        self._repository = repository

    async def execute(self, data: CreateRepairRequestInput) -> CreateRepairRequestResult:
        request_id = str(uuid4())
        request = RepairRequest.new(
            request_id=request_id,
            chat_id=data.chat_id,
            user_id=data.user_id,
            mode=data.mode,
        )
        await self._repository.add(request)
        logger.info("Created repair_request id={} mode={} status={}", request.id, request.mode, request.status)
        return CreateRepairRequestResult(request=request)


@dataclass
class UploadPhotoInput:
    request_id: str
    image_key: str


@dataclass
class UploadPhotoResult:
    request: RepairRequest


class UploadPhotoUseCase:
    def __init__(self, repository: RepairRequestRepository, state_machine: RequestStateMachine) -> None:
        self._repository = repository
        self._state_machine = state_machine

    async def execute(self, data: UploadPhotoInput) -> UploadPhotoResult:
        request = await self._repository.get(data.request_id)
        if request is None:
            raise ValueError(f"request {data.request_id} not found")
        if request.mode is RequestMode.MANUAL:
            raise ValueError("photo upload is not supported for manual mode")
        if request.status is not RequestStatus.CREATED:
            raise ValueError("photo can only be uploaded for requests in CREATED status")

        with_image = RepairRequest(
            id=request.id,
            chat_id=request.chat_id,
            user_id=request.user_id,
            mode=request.mode,
            status=request.status,
            created_at=request.created_at,
            updated_at=request.updated_at,
            timeout_at=request.timeout_at,
            original_image_key=data.image_key,
            composited_image_key=request.composited_image_key,
        )
        queued = self._state_machine.transition(with_image, RequestStatus.QUEUED)
        await self._repository.update(queued)
        logger.info("Uploaded photo for request id={} image_key={}", queued.id, data.image_key)
        return UploadPhotoResult(request=queued)


@dataclass
class ConfirmPricingInput:
    request_id: str


@dataclass
class ConfirmPricingResult:
    request: RepairRequest
    total_cost: float
    total_hours: float


class ConfirmPricingUseCase:
    def __init__(self, repository: RepairRequestRepository, state_machine: RequestStateMachine) -> None:
        self._repository = repository
        self._state_machine = state_machine

    async def execute(self, data: ConfirmPricingInput) -> ConfirmPricingResult:
        request = await self._repository.get(data.request_id)
        if request is None:
            raise ValueError(f"request {data.request_id} not found")
        if request.status is not RequestStatus.PRICING:
            raise ValueError("pricing can only be confirmed from PRICING status")

        done = self._state_machine.transition(request, RequestStatus.DONE)
        await self._repository.update(done)
        logger.info("Confirmed pricing for request id={}", done.id)
        return ConfirmPricingResult(request=done, total_cost=0.0, total_hours=0.0)
