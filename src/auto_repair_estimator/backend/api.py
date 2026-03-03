from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
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


router = APIRouter(prefix="/v1/requests", tags=["requests"])

_repository = InMemoryRepairRequestRepository()
_state_machine = RequestStateMachine()


def get_repository() -> InMemoryRepairRequestRepository:
    return _repository


def get_state_machine() -> RequestStateMachine:
    return _state_machine


def get_create_use_case(
    repository: InMemoryRepairRequestRepository = Depends(get_repository),
) -> CreateRepairRequestUseCase:
    return CreateRepairRequestUseCase(repository=repository)


def get_upload_photo_use_case(
    repository: InMemoryRepairRequestRepository = Depends(get_repository),
    state_machine: RequestStateMachine = Depends(get_state_machine),
) -> UploadPhotoUseCase:
    return UploadPhotoUseCase(repository=repository, state_machine=state_machine)


def get_confirm_pricing_use_case(
    repository: InMemoryRepairRequestRepository = Depends(get_repository),
    state_machine: RequestStateMachine = Depends(get_state_machine),
) -> ConfirmPricingUseCase:
    return ConfirmPricingUseCase(repository=repository, state_machine=state_machine)


class CreateRequestBody(BaseModel):
    telegram_chat_id: int
    telegram_user_id: int | None = None
    mode: RequestMode


class RequestResponse(BaseModel):
    id: str
    status: RequestStatus
    mode: RequestMode


class UploadPhotoBody(BaseModel):
    image_key: str


class PricingResponse(BaseModel):
    id: str
    status: RequestStatus
    total_cost: float
    total_hours: float


@router.post("", response_model=RequestResponse)
async def create_request(
    body: CreateRequestBody,
    use_case: CreateRepairRequestUseCase = Depends(get_create_use_case),
) -> RequestResponse:
    result = await use_case.execute(
        CreateRepairRequestInput(
            telegram_chat_id=body.telegram_chat_id,
            telegram_user_id=body.telegram_user_id,
            mode=body.mode,
        )
    )
    return RequestResponse(id=result.request.id, status=result.request.status, mode=result.request.mode)


@router.post("/{request_id}/photo", response_model=RequestResponse)
async def upload_photo(
    request_id: str,
    body: UploadPhotoBody,
    use_case: UploadPhotoUseCase = Depends(get_upload_photo_use_case),
) -> RequestResponse:
    try:
        result = await use_case.execute(UploadPhotoInput(request_id=request_id, image_key=body.image_key))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RequestResponse(id=result.request.id, status=result.request.status, mode=result.request.mode)


@router.post("/{request_id}/confirm-pricing", response_model=PricingResponse)
async def confirm_pricing(
    request_id: str,
    use_case: ConfirmPricingUseCase = Depends(get_confirm_pricing_use_case),
) -> PricingResponse:
    try:
        result = await use_case.execute(ConfirmPricingInput(request_id=request_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PricingResponse(
        id=result.request.id,
        status=result.request.status,
        total_cost=result.total_cost,
        total_hours=result.total_hours,
    )

