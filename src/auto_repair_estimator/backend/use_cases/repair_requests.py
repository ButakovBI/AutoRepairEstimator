from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger

from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.outbox_repository import OutboxRepository
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


@dataclass
class CreateRepairRequestInput:
    chat_id: int
    user_id: int | None
    mode: RequestMode
    idempotency_key: str | None = None


@dataclass
class CreateRepairRequestResult:
    request: RepairRequest


class CreateRepairRequestUseCase:
    def __init__(self, repository: RepairRequestRepository) -> None:
        self._repository = repository

    async def execute(self, data: CreateRepairRequestInput) -> CreateRepairRequestResult:
        # If the caller provided an idempotency key, consult the repo first so
        # a retried VK photo event doesn't spawn a second request row. The
        # repository layer handles the UNIQUE violation as well (second line
        # of defence), but checking up-front avoids a noisy DB error.
        if data.idempotency_key is not None:
            existing = await self._repository.get_by_idempotency_key(data.idempotency_key)
            if existing is not None:
                logger.info(
                    "CreateRepairRequest deduplicated id={} idempotency_key={}",
                    existing.id,
                    data.idempotency_key,
                )
                return CreateRepairRequestResult(request=existing)

        request_id = str(uuid4())
        request = RepairRequest.new(
            request_id=request_id,
            chat_id=data.chat_id,
            user_id=data.user_id,
            mode=data.mode,
            idempotency_key=data.idempotency_key,
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
    """Marks a ML request as QUEUED and enqueues an ``inference_requests`` outbox event.

    The DB update and the outbox insert are meant to run in the same Postgres
    transaction at the adapter level. Here we keep both writes through the
    repository abstractions so tests can use in-memory fakes.
    """

    def __init__(
        self,
        repository: RepairRequestRepository,
        state_machine: RequestStateMachine,
        outbox_repository: OutboxRepository,
        inference_requests_topic: str,
    ) -> None:
        self._repository = repository
        self._state_machine = state_machine
        self._outbox = outbox_repository
        self._inference_requests_topic = inference_requests_topic

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
            idempotency_key=request.idempotency_key,
        )
        queued = self._state_machine.transition(with_image, RequestStatus.QUEUED)
        await self._repository.update(queued)

        event = OutboxEvent(
            id=str(uuid4()),
            aggregate_id=queued.id,
            topic=self._inference_requests_topic,
            payload={"request_id": queued.id, "image_key": data.image_key},
            created_at=datetime.now(UTC),
        )
        # Transactional-outbox contract: the request is QUEUED iff the
        # matching outbox_event is persisted. We don't have a shared
        # transaction handle across both repositories in this layer, so if
        # the outbox insert fails we explicitly roll the request back to
        # its prior state (CREATED, no image_key). In the Postgres adapter
        # both writes should run inside one asyncpg transaction for real
        # atomicity — this rollback is the fallback guarantee.
        try:
            await self._outbox.add(event)
        except Exception:
            logger.error(
                "Outbox insert failed for request id={} — rolling back QUEUED -> CREATED",
                queued.id,
            )
            await self._repository.update(request)
            raise

        logger.info(
            "Uploaded photo for request id={} image_key={} -> outbox inference_requests", queued.id, data.image_key
        )
        return UploadPhotoResult(request=queued)


@dataclass
class ConfirmPricingInput:
    request_id: str


@dataclass
class ConfirmPricingResult:
    request: RepairRequest


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
        return ConfirmPricingResult(request=done)


@dataclass
class AbandonRequestInput:
    request_id: str
    # Cause string is stored in ``ml_error_code`` so operators and the
    # heartbeat watchdog can tell a user-initiated abandon apart from a
    # real ML failure or an actual timeout. Defaults to the generic
    # "user_abandoned" marker.
    reason: str = "user_abandoned"


@dataclass
class AbandonRequestResult:
    request: RepairRequest
    was_terminal: bool  # True if the caller's request was already DONE/FAILED.


class AbandonRequestUseCase:
    """Explicitly abandon a user's active session.

    The bot calls this whenever the user presses "Начать" or picks a new
    mode while an older request is still non-terminal — without this
    primitive, each such click silently leaves an orphaned row behind and
    lets ``get_latest_active_by_chat_id`` flip between sessions in
    surprising ways. The use case is idempotent: calling it on an
    already-DONE/FAILED request is a no-op that returns the current state.
    """

    def __init__(
        self,
        repository: RepairRequestRepository,
        state_machine: RequestStateMachine,
    ) -> None:
        self._repository = repository
        self._state_machine = state_machine

    async def execute(self, data: AbandonRequestInput) -> AbandonRequestResult:
        request = await self._repository.get(data.request_id)
        if request is None:
            raise ValueError(f"request {data.request_id} not found")
        if request.status in {RequestStatus.DONE, RequestStatus.FAILED}:
            logger.info(
                "AbandonRequest no-op id={} already_status={}",
                request.id,
                request.status.value,
            )
            return AbandonRequestResult(request=request, was_terminal=True)

        failed = self._state_machine.transition(request, RequestStatus.FAILED)
        # Stamp the cause so diagnostics can distinguish "user explicitly
        # restarted" from "watchdog timed the session out".
        failed = RepairRequest(
            id=failed.id,
            chat_id=failed.chat_id,
            user_id=failed.user_id,
            mode=failed.mode,
            status=failed.status,
            created_at=failed.created_at,
            updated_at=failed.updated_at,
            timeout_at=failed.timeout_at,
            original_image_key=failed.original_image_key,
            composited_image_key=failed.composited_image_key,
            ml_error_code=data.reason,
            ml_error_message=failed.ml_error_message,
            idempotency_key=failed.idempotency_key,
        )
        await self._repository.update(failed)
        logger.info(
            "AbandonRequest id={} prev_status={} reason={}",
            failed.id,
            request.status.value,
            data.reason,
        )
        return AbandonRequestResult(request=failed, was_terminal=False)
