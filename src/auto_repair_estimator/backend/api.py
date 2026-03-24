"""FastAPI router. All infrastructure is injected via request.app.state so tests
can substitute any repository implementation without touching module globals."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auto_repair_estimator.backend.domain.interfaces.damage_repository import DamageRepository
from auto_repair_estimator.backend.domain.interfaces.pricing_rule_repository import PricingRuleRepository
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.services.pricing_service import PricingService
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageType,
    PartType,
    RequestMode,
    RequestStatus,
)
from auto_repair_estimator.backend.use_cases.calculate_pricing import CalculatePricingInput, CalculatePricingUseCase
from auto_repair_estimator.backend.use_cases.manage_damages import (
    AddDamageInput,
    AddDamageUseCase,
    DeleteDamageInput,
    DeleteDamageUseCase,
    EditDamageInput,
    EditDamageUseCase,
)
from auto_repair_estimator.backend.use_cases.repair_requests import (
    ConfirmPricingInput,
    ConfirmPricingUseCase,
    CreateRepairRequestInput,
    CreateRepairRequestUseCase,
    UploadPhotoInput,
    UploadPhotoUseCase,
)

router = APIRouter(prefix="/v1/requests", tags=["requests"])

_state_machine = RequestStateMachine()


# ---------------------------------------------------------------------------
# Dependency helpers — read from app.state so tests can inject any impl
# ---------------------------------------------------------------------------


def _request_repo(request: Request) -> RepairRequestRepository:
    return request.app.state.request_repo  # type: ignore[no-any-return]


def _damage_repo(request: Request) -> DamageRepository:
    return request.app.state.damage_repo  # type: ignore[no-any-return]


def _pricing_rule_repo(request: Request) -> PricingRuleRepository:
    return request.app.state.pricing_rule_repo  # type: ignore[no-any-return]


def _sm(request: Request) -> RequestStateMachine:
    return _state_machine


def get_create_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
) -> CreateRepairRequestUseCase:
    return CreateRepairRequestUseCase(repository=repo)


def get_upload_photo_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
    sm: RequestStateMachine = Depends(_sm),
) -> UploadPhotoUseCase:
    return UploadPhotoUseCase(repository=repo, state_machine=sm)


def get_confirm_pricing_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
    sm: RequestStateMachine = Depends(_sm),
) -> ConfirmPricingUseCase:
    return ConfirmPricingUseCase(repository=repo, state_machine=sm)


def get_add_damage_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
    dmg_repo: DamageRepository = Depends(_damage_repo),
) -> AddDamageUseCase:
    return AddDamageUseCase(request_repository=repo, damage_repository=dmg_repo)


def get_edit_damage_use_case(
    dmg_repo: DamageRepository = Depends(_damage_repo),
) -> EditDamageUseCase:
    return EditDamageUseCase(damage_repository=dmg_repo)


def get_delete_damage_use_case(
    dmg_repo: DamageRepository = Depends(_damage_repo),
) -> DeleteDamageUseCase:
    return DeleteDamageUseCase(damage_repository=dmg_repo)


def get_calculate_pricing_use_case(
    dmg_repo: DamageRepository = Depends(_damage_repo),
    rule_repo: PricingRuleRepository = Depends(_pricing_rule_repo),
) -> CalculatePricingUseCase:
    return CalculatePricingUseCase(
        damage_repository=dmg_repo,
        pricing_service=PricingService(_rule_repository=rule_repo),
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CreateRequestBody(BaseModel):
    chat_id: int
    user_id: int | None = None
    mode: RequestMode


class RequestResponse(BaseModel):
    id: str
    status: RequestStatus
    mode: RequestMode
    presigned_put_url: str | None = None


class UploadPhotoBody(BaseModel):
    image_key: str


class DamageBody(BaseModel):
    part_type: PartType
    damage_type: DamageType


class EditDamageBody(BaseModel):
    damage_type: DamageType


class DamageResponse(BaseModel):
    id: str
    part_type: PartType
    damage_type: DamageType
    source: str
    is_deleted: bool


class PricingResponse(BaseModel):
    id: str
    status: RequestStatus
    total_cost: float
    total_hours: float
    breakdown: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{request_id}", response_model=dict)
async def get_request(
    request_id: str,
    repo: RepairRequestRepository = Depends(_request_repo),
    dmg_repo: DamageRepository = Depends(_damage_repo),
) -> dict:  # type: ignore[type-arg]
    request = await repo.get(request_id)
    if request is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    damages = await dmg_repo.get_by_request_id(request_id)
    return {
        "id": request.id,
        "status": request.status.value,
        "mode": request.mode.value,
        "damages": [
            {
                "id": d.id,
                "damage_type": d.damage_type.value,
                "part_type": d.part_type.value,
                "source": d.source.value,
                "is_deleted": d.is_deleted,
            }
            for d in damages
        ],
    }


@router.post("", response_model=RequestResponse)
async def create_request(
    body: CreateRequestBody,
    use_case: CreateRepairRequestUseCase = Depends(get_create_use_case),
) -> RequestResponse:
    result = await use_case.execute(
        CreateRepairRequestInput(
            chat_id=body.chat_id,
            user_id=body.user_id,
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


@router.post("/{request_id}/damages", response_model=DamageResponse)
async def add_damage(
    request_id: str,
    body: DamageBody,
    use_case: AddDamageUseCase = Depends(get_add_damage_use_case),
) -> DamageResponse:
    try:
        result = await use_case.execute(
            AddDamageInput(request_id=request_id, part_type=body.part_type, damage_type=body.damage_type)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    d = result.damage
    return DamageResponse(
        id=d.id, part_type=d.part_type, damage_type=d.damage_type, source=d.source.value, is_deleted=d.is_deleted
    )


@router.patch("/{request_id}/damages/{damage_id}", response_model=DamageResponse)
async def edit_damage(
    request_id: str,
    damage_id: str,
    body: EditDamageBody,
    use_case: EditDamageUseCase = Depends(get_edit_damage_use_case),
) -> DamageResponse:
    try:
        result = await use_case.execute(EditDamageInput(damage_id=damage_id, damage_type=body.damage_type))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    d = result.damage
    return DamageResponse(
        id=d.id, part_type=d.part_type, damage_type=d.damage_type, source=d.source.value, is_deleted=d.is_deleted
    )


@router.delete("/{request_id}/damages/{damage_id}", status_code=204, response_model=None)
async def delete_damage(
    request_id: str,
    damage_id: str,
    use_case: DeleteDamageUseCase = Depends(get_delete_damage_use_case),
) -> None:
    try:
        await use_case.execute(DeleteDamageInput(damage_id=damage_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{request_id}/confirm", response_model=PricingResponse)
async def confirm_pricing(
    request_id: str,
    confirm_uc: ConfirmPricingUseCase = Depends(get_confirm_pricing_use_case),
    pricing_uc: CalculatePricingUseCase = Depends(get_calculate_pricing_use_case),
) -> PricingResponse:
    pricing_result = await pricing_uc.execute(CalculatePricingInput(request_id=request_id))
    try:
        result = await confirm_uc.execute(ConfirmPricingInput(request_id=request_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PricingResponse(
        id=result.request.id,
        status=result.request.status,
        total_cost=pricing_result.total_cost,
        total_hours=pricing_result.total_hours,
        breakdown=pricing_result.breakdown,
    )


@router.post("/{request_id}/confirm-pricing", response_model=PricingResponse)
async def confirm_pricing_legacy(
    request_id: str,
    confirm_uc: ConfirmPricingUseCase = Depends(get_confirm_pricing_use_case),
    pricing_uc: CalculatePricingUseCase = Depends(get_calculate_pricing_use_case),
) -> PricingResponse:
    return await confirm_pricing(request_id, confirm_uc, pricing_uc)
