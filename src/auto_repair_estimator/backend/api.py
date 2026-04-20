"""FastAPI router. All infrastructure is injected via request.app.state so tests
can substitute any repository implementation without touching module globals."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auto_repair_estimator.backend.domain.interfaces.damage_repository import DamageRepository
from auto_repair_estimator.backend.domain.interfaces.outbox_repository import OutboxRepository
from auto_repair_estimator.backend.domain.interfaces.pricing_rule_repository import PricingRuleRepository
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.interfaces.storage_gateway import StorageGateway
from auto_repair_estimator.backend.domain.services.image_validator import validate_image_bytes
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
    AbandonRequestInput,
    AbandonRequestUseCase,
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


def _storage(request: Request) -> StorageGateway | None:
    """Return the configured storage gateway or ``None`` if the app runs in
    dev/test mode without real object storage. Endpoints gracefully skip
    bytes-level validation in that case.
    """

    return getattr(request.app.state, "storage", None)


def _outbox_repo(request: Request) -> OutboxRepository:
    return request.app.state.outbox_repo  # type: ignore[no-any-return]


def _raw_bucket(request: Request) -> str:
    return getattr(request.app.state, "s3_bucket_raw", "raw-images")  # type: ignore[no-any-return]


def _inference_requests_topic(request: Request) -> str:
    return getattr(request.app.state, "kafka_topic_inference_requests", "inference_requests")  # type: ignore[no-any-return]


def _sm(request: Request) -> RequestStateMachine:
    return _state_machine


def get_create_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
) -> CreateRepairRequestUseCase:
    return CreateRepairRequestUseCase(repository=repo)


def get_upload_photo_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
    sm: RequestStateMachine = Depends(_sm),
    outbox: OutboxRepository = Depends(_outbox_repo),
    inference_requests_topic: str = Depends(_inference_requests_topic),
) -> UploadPhotoUseCase:
    return UploadPhotoUseCase(
        repository=repo,
        state_machine=sm,
        outbox_repository=outbox,
        inference_requests_topic=inference_requests_topic,
    )


def get_confirm_pricing_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
    sm: RequestStateMachine = Depends(_sm),
) -> ConfirmPricingUseCase:
    return ConfirmPricingUseCase(repository=repo, state_machine=sm)


def get_abandon_request_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
    sm: RequestStateMachine = Depends(_sm),
) -> AbandonRequestUseCase:
    return AbandonRequestUseCase(repository=repo, state_machine=sm)


def get_add_damage_use_case(
    repo: RepairRequestRepository = Depends(_request_repo),
    dmg_repo: DamageRepository = Depends(_damage_repo),
) -> AddDamageUseCase:
    return AddDamageUseCase(request_repository=repo, damage_repository=dmg_repo)


def get_edit_damage_use_case(
    dmg_repo: DamageRepository = Depends(_damage_repo),
    req_repo: RepairRequestRepository = Depends(_request_repo),
) -> EditDamageUseCase:
    return EditDamageUseCase(damage_repository=dmg_repo, request_repository=req_repo)


def get_delete_damage_use_case(
    dmg_repo: DamageRepository = Depends(_damage_repo),
    req_repo: RepairRequestRepository = Depends(_request_repo),
) -> DeleteDamageUseCase:
    return DeleteDamageUseCase(damage_repository=dmg_repo, request_repository=req_repo)


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
    idempotency_key: str | None = None


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
    part_type: PartType | None = None


class DamageResponse(BaseModel):
    id: str
    part_type: PartType
    damage_type: DamageType
    source: str
    is_deleted: bool


class PricingResponse(BaseModel):
    id: str
    status: RequestStatus
    total_cost_min: float
    total_cost_max: float
    total_hours_min: float
    total_hours_max: float
    breakdown: list[dict[str, Any]]
    notes: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class ActiveRequestResponse(BaseModel):
    """Bot-facing summary of a user's latest non-terminal session."""

    id: str
    status: RequestStatus
    mode: RequestMode
    chat_id: int


# IMPORTANT: this endpoint must be declared *before* ``/{request_id}``,
# otherwise FastAPI greedily matches ``/active`` as a request_id path
# parameter and routes every lookup into ``get_request`` (which then 404s).
@router.get("/active", response_model=ActiveRequestResponse)
async def get_active_request(
    chat_id: int,
    repo: RepairRequestRepository = Depends(_request_repo),
) -> ActiveRequestResponse:
    """Return the latest non-terminal request for ``chat_id`` or 404."""
    request = await repo.get_latest_active_by_chat_id(chat_id)
    if request is None:
        raise HTTPException(status_code=404, detail=f"No active request for chat_id={chat_id}")
    return ActiveRequestResponse(
        id=request.id,
        status=request.status,
        mode=request.mode,
        chat_id=request.chat_id,
    )


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
    storage: StorageGateway | None = Depends(_storage),
    raw_bucket: str = Depends(_raw_bucket),
) -> RequestResponse:
    result = await use_case.execute(
        CreateRepairRequestInput(
            chat_id=body.chat_id,
            user_id=body.user_id,
            mode=body.mode,
            idempotency_key=body.idempotency_key,
        )
    )

    presigned_url: str | None = None
    if body.mode is RequestMode.ML and storage is not None:
        image_key = f"{raw_bucket}/{result.request.id}.jpg"
        presigned_url = await storage.generate_presigned_put_url(image_key)

    return RequestResponse(
        id=result.request.id,
        status=result.request.status,
        mode=result.request.mode,
        presigned_put_url=presigned_url,
    )


@router.post("/{request_id}/photo", response_model=RequestResponse)
async def upload_photo(
    request_id: str,
    body: UploadPhotoBody,
    use_case: UploadPhotoUseCase = Depends(get_upload_photo_use_case),
    storage: StorageGateway | None = Depends(_storage),
) -> RequestResponse:
    # Bytes-level validation runs only when real object storage is wired in;
    # in-memory dev/test mode skips it (no bucket to download from).
    if storage is not None:
        try:
            data = await storage.download(body.image_key)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"image not found in storage: {exc}") from exc
        try:
            validate_image_bytes(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        result = await use_case.execute(
            EditDamageInput(
                damage_id=damage_id,
                damage_type=body.damage_type,
                part_type=body.part_type,
                request_id=request_id,
            )
        )
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
        await use_case.execute(DeleteDamageInput(damage_id=damage_id, request_id=request_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        total_cost_min=pricing_result.total_cost_min,
        total_cost_max=pricing_result.total_cost_max,
        total_hours_min=pricing_result.total_hours_min,
        total_hours_max=pricing_result.total_hours_max,
        breakdown=pricing_result.breakdown,
        notes=pricing_result.notes,
    )


class AbandonRequestResponse(BaseModel):
    id: str
    status: RequestStatus
    was_already_terminal: bool


@router.post("/{request_id}/abandon", response_model=AbandonRequestResponse)
async def abandon_request(
    request_id: str,
    use_case: AbandonRequestUseCase = Depends(get_abandon_request_use_case),
) -> AbandonRequestResponse:
    """Transition a non-terminal request to FAILED with a user-abandon marker.

    Idempotent — calling on an already-DONE/FAILED request returns the
    current state with ``was_already_terminal=True`` instead of erroring.
    The bot relies on this so it can always call ``abandon_request`` before
    starting a new session for the same chat, without branching on the
    existing status.
    """
    try:
        result = await use_case.execute(AbandonRequestInput(request_id=request_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AbandonRequestResponse(
        id=result.request.id,
        status=result.request.status,
        was_already_terminal=result.was_terminal,
    )


@router.post("/{request_id}/confirm-pricing", response_model=PricingResponse)
async def confirm_pricing_legacy(
    request_id: str,
    confirm_uc: ConfirmPricingUseCase = Depends(get_confirm_pricing_use_case),
    pricing_uc: CalculatePricingUseCase = Depends(get_calculate_pricing_use_case),
) -> PricingResponse:
    return await confirm_pricing(request_id, confirm_uc, pricing_uc)
