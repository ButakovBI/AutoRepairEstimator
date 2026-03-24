from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.detected_part import DetectedPart
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.damage_repository import DamageRepository
from auto_repair_estimator.backend.domain.interfaces.outbox_repository import OutboxRepository
from auto_repair_estimator.backend.domain.interfaces.part_repository import PartRepository
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
    RequestStatus,
)


@dataclass
class InferencePartData:
    part_type: str
    confidence: float
    bbox: list[float]
    crop_image_key: str | None


@dataclass
class InferenceDamageData:
    damage_type: str
    part_type: str
    confidence: float
    mask_image_key: str | None


@dataclass
class ProcessInferenceResultInput:
    request_id: str
    status: str
    parts: list[InferencePartData]
    damages: list[InferenceDamageData]
    composited_image_key: str | None
    error_message: str | None


class ProcessInferenceResultUseCase:
    def __init__(
        self,
        request_repository: RepairRequestRepository,
        part_repository: PartRepository,
        damage_repository: DamageRepository,
        outbox_repository: OutboxRepository,
        state_machine: RequestStateMachine,
        notifications_topic: str,
    ) -> None:
        self._requests = request_repository
        self._parts = part_repository
        self._damages = damage_repository
        self._outbox = outbox_repository
        self._sm = state_machine
        self._notifications_topic = notifications_topic

    async def execute(self, data: ProcessInferenceResultInput) -> None:
        request = await self._requests.get(data.request_id)
        if request is None:
            logger.warning("ProcessInferenceResult: request {} not found", data.request_id)
            return

        if request.status is RequestStatus.PRICING:
            logger.info("ProcessInferenceResult: request {} already in PRICING, ignoring duplicate", data.request_id)
            return

        is_success = data.status == "success"
        saved_parts: list[DetectedPart] = []

        if is_success:
            saved_parts = await self._save_parts(data)
            part_by_type = {p.part_type.value: p for p in saved_parts}
            await self._save_damages(data, part_by_type)

        updated_request = await self._update_request_status(request, data)
        await self._create_notification_event(updated_request, data, is_success)

        logger.info(
            "Processed inference result for request={} status={} parts={} damages={}",
            data.request_id,
            data.status,
            len(data.parts),
            len(data.damages),
        )

    async def _save_parts(self, data: ProcessInferenceResultInput) -> list[DetectedPart]:
        parts: list[DetectedPart] = []
        for p in data.parts:
            try:
                bbox = p.bbox if len(p.bbox) >= 4 else [0.0, 0.0, 0.0, 0.0]
                part = DetectedPart(
                    id=str(uuid4()),
                    request_id=data.request_id,
                    part_type=PartType(p.part_type),
                    confidence=p.confidence,
                    x=bbox[0],
                    y=bbox[1],
                    width=bbox[2],
                    height=bbox[3],
                    crop_image_key=p.crop_image_key,
                )
                await self._parts.add(part)
                parts.append(part)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping unknown part_type={}: {}", p.part_type, exc)
        return parts

    async def _save_damages(self, data: ProcessInferenceResultInput, part_by_type: dict[str, DetectedPart]) -> None:
        for d in data.damages:
            try:
                linked_part = part_by_type.get(d.part_type)
                damage = DetectedDamage(
                    id=str(uuid4()),
                    request_id=data.request_id,
                    damage_type=DamageType(d.damage_type),
                    part_type=PartType(d.part_type),
                    source=DamageSource.ML,
                    is_deleted=False,
                    part_id=linked_part.id if linked_part else None,
                    confidence=d.confidence,
                    mask_image_key=d.mask_image_key,
                )
                await self._damages.add(damage)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping unknown damage_type={} or part_type={}: {}", d.damage_type, d.part_type, exc)

    async def _update_request_status(self, request: RepairRequest, data: ProcessInferenceResultInput) -> RepairRequest:
        updated = RepairRequest(
            id=request.id,
            chat_id=request.chat_id,
            user_id=request.user_id,
            mode=request.mode,
            status=request.status,
            created_at=request.created_at,
            updated_at=request.updated_at,
            timeout_at=request.timeout_at,
            original_image_key=request.original_image_key,
            composited_image_key=data.composited_image_key,
        )
        priced = self._sm.transition(updated, RequestStatus.PRICING)
        await self._requests.update(priced)
        return priced

    async def _create_notification_event(
        self, request: RepairRequest, data: ProcessInferenceResultInput, is_success: bool
    ) -> None:
        notification_type = "inference_complete" if is_success else "inference_failed"
        payload: dict[str, Any] = {
            "chat_id": request.chat_id,
            "request_id": request.id,
            "type": notification_type,
            "composited_image_key": data.composited_image_key,
            "damages": [{"damage_type": d.damage_type, "part_type": d.part_type} for d in data.damages],
        }
        event = OutboxEvent(
            id=str(uuid4()),
            aggregate_id=request.id,
            topic=self._notifications_topic,
            payload=payload,
            created_at=datetime.now(UTC),
        )
        await self._outbox.add(event)
