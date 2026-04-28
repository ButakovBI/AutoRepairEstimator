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
    RequestMode,
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

        if request.status in {RequestStatus.PRICING, RequestStatus.DONE, RequestStatus.FAILED}:
            logger.info(
                "ProcessInferenceResult: request {} is in terminal/advanced state {}, ignoring duplicate",
                data.request_id,
                request.status.value,
            )
            return

        # CREATED means the upload-photo step has not been run yet; the ML
        # Worker could not possibly have produced a result for it. Accepting
        # this message would use the MANUAL-only CREATED->PRICING edge, which
        # is wrong for an ML request (it would skip QUEUED/PROCESSING).
        if request.status is RequestStatus.CREATED:
            logger.warning(
                "ProcessInferenceResult: request {} is still CREATED — ignoring "
                "stale/out-of-order inference_results message",
                data.request_id,
            )
            return

        # MANUAL requests never participate in the Kafka pipeline; a result
        # for one is a cross-mode bug or a replay. Drop it so manual damages
        # are not overwritten with ML predictions.
        if request.mode is RequestMode.MANUAL:
            logger.warning(
                "ProcessInferenceResult: request {} is MANUAL mode — ignoring",
                data.request_id,
            )
            return

        # Idempotency for PROCESSING state: if the previous delivery already
        # persisted detected parts for this request but crashed before moving
        # the state machine to PRICING, a redelivered message must NOT create
        # duplicate parts/damages. We detect the "already processed" condition
        # by checking for pre-existing detected parts; a fresh request cannot
        # have any.
        if request.status is RequestStatus.PROCESSING:
            existing_parts = await self._parts.get_by_request_id(data.request_id)
            if existing_parts:
                logger.info(
                    "ProcessInferenceResult: request {} already has {} detected parts; "
                    "finishing transition to PRICING without duplicating data",
                    data.request_id,
                    len(existing_parts),
                )
                # Rebuild the notification payload from the damages
                # actually persisted during the prior partial run —
                # echoing ``data.damages`` verbatim would leak the raw
                # (duplicate-heavy) detector output into the user's view
                # and desync from what the edit screen will show.
                existing_damages = await self._damages.get_by_request_id(data.request_id)
                replayed_pairs = [
                    (d.part_type.value, d.damage_type.value)
                    for d in existing_damages
                    if not d.is_deleted
                ]
                updated_request = await self._update_request_status(request, data)
                await self._create_notification_event(
                    updated_request,
                    data,
                    is_success=data.status == "success",
                    persisted_damage_pairs=replayed_pairs,
                )
                return

        if request.status is RequestStatus.QUEUED:
            request = self._sm.transition(request, RequestStatus.PROCESSING)
            await self._requests.update(request)

        is_success = data.status == "success"
        saved_parts: list[DetectedPart] = []
        persisted_damage_pairs: list[tuple[str, str]] = []

        if is_success:
            saved_parts = await self._save_parts(data)
            part_by_type = {p.part_type.value: p for p in saved_parts}
            persisted_damage_pairs = await self._save_damages(data, part_by_type)

        updated_request = await self._update_request_status(request, data)
        # Pass the post-dedup, post-persist list of pairs to the notification
        # layer so the "Обнаруженные повреждения" card the user sees in VK
        # is identical to what the backend actually stored. Before this
        # change the notification echoed the raw detector output (26
        # "Бампер — Царапина" from a single scratched bumper), which both
        # spammed the chat and desynced from the edit screen.
        await self._create_notification_event(
            updated_request, data, is_success, persisted_damage_pairs
        )

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

    async def _save_damages(
        self, data: ProcessInferenceResultInput, part_by_type: dict[str, DetectedPart]
    ) -> list[tuple[str, str]]:
        """Persist ML damages with ``(part_type, damage_type)`` uniqueness.

        The detector frequently emits multiple masks for the same class on
        the same part (17 scratches on one door, 12 on a bumper). Saving
        each one separately spams the user's edit screen and forces the
        pricing aggregator to collapse them later anyway — the business
        rule is "one painting per painted part, one replacement per
        replaced part", not "one priced row per mask". We enforce the
        uniqueness invariant at ingestion, the earliest surface where it
        can be expressed, so every downstream surface (edit UI, pricing,
        notifications) sees a clean, deduplicated basket.

        The first detection in iteration order wins. Detection order is
        effectively confidence-sorted coming out of the ML worker, so the
        highest-confidence representative survives — this matches the
        `damage_aggregator` tie-break and keeps audit logs stable.

        Returns the ordered ``(part_type, damage_type)`` pairs that were
        actually persisted so the caller can forward them to the
        notification payload.
        """

        seen: set[tuple[str, str]] = set()
        duplicates_dropped = 0
        persisted_pairs: list[tuple[str, str]] = []
        for d in data.damages:
            key = (d.part_type, d.damage_type)
            if key in seen:
                duplicates_dropped += 1
                continue
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
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Skipping unknown damage_type={} or part_type={}: {}",
                    d.damage_type,
                    d.part_type,
                    exc,
                )
                continue
            await self._damages.add(damage)
            seen.add(key)
            persisted_pairs.append(key)

        if duplicates_dropped:
            logger.info(
                "ProcessInferenceResult: collapsed {} duplicate (part, damage) "
                "pairs for request={} before persistence",
                duplicates_dropped,
                data.request_id,
            )
        return persisted_pairs

    async def _update_request_status(self, request: RepairRequest, data: ProcessInferenceResultInput) -> RepairRequest:
        # Stamp the worker's ``error_message`` on the request on the
        # failure branch so the reason is recoverable from the database
        # alone (no need to cross-reference Kafka logs). On the success
        # branch we intentionally leave the field empty — a prior
        # failure on the same id should not linger after we successfully
        # re-ran inference for a different photo.
        failed_branch = data.status != "success"
        error_code = "inference_failed" if failed_branch else None
        error_message = data.error_message if failed_branch else None
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
            ml_error_code=error_code,
            ml_error_message=error_message,
            idempotency_key=request.idempotency_key,
        )
        priced = self._sm.transition(updated, RequestStatus.PRICING)
        await self._requests.update(priced)
        return priced

    async def _create_notification_event(
        self,
        request: RepairRequest,
        data: ProcessInferenceResultInput,
        is_success: bool,
        persisted_damage_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        notification_type = "inference_complete" if is_success else "inference_failed"
        # Prefer the post-dedup list — that's what the DB actually holds,
        # and what the edit screen will show. Fall back to the raw
        # detector output on the failure branch (no persistence happened
        # there) so we still have something for the bot to log.
        if persisted_damage_pairs is not None:
            notification_damages = [
                {"damage_type": dt, "part_type": pt}
                for pt, dt in persisted_damage_pairs
            ]
        else:
            notification_damages = [
                {"damage_type": d.damage_type, "part_type": d.part_type}
                for d in data.damages
            ]
        payload: dict[str, Any] = {
            "chat_id": request.chat_id,
            "request_id": request.id,
            "type": notification_type,
            "composited_image_key": data.composited_image_key,
            "damages": notification_damages,
        }
        # For the failure branch the worker sends a short ``error_message``
        # string ("no_parts_detected", "inference_failed", ...) — pipe it
        # through to the bot so the user sees a specific hint instead of
        # a generic "ML не справилось". On the success branch this field
        # stays absent and the bot renders the damages list as before.
        if not is_success and data.error_message:
            payload["error_message"] = data.error_message
        event = OutboxEvent(
            id=str(uuid4()),
            aggregate_id=request.id,
            topic=self._notifications_topic,
            payload=payload,
            created_at=datetime.now(UTC),
        )
        await self._outbox.add(event)
