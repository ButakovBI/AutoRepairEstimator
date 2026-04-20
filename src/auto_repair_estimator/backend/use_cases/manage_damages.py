"""Damage-mutation use cases (Add / Edit / Delete).

All three follow the same lifecycle contract:

1. The parent ``RepairRequest`` must exist and must be in ``PRICING`` —
   this is the only state where the user is editing the basket of
   damages. Every other status (``CREATED``, ``QUEUED``, ``PROCESSING``,
   ``DONE``, ``FAILED``) rejects the mutation with ``ValueError``.
2. For ``EditDamage`` / ``DeleteDamage`` the target damage must belong
   to the same ``request_id`` passed in the input — this prevents a
   client that happens to know a damage UUID from tampering with
   another user's session (cross-request isolation).
3. On success the parent request's ``timeout_at`` is pushed to
   ``now + 5 min`` so the HeartbeatWatchdog doesn't kill a session where
   the user is clearly still engaged.

``EditDamageUseCase`` / ``DeleteDamageUseCase`` take ``request_repository``
as an *optional* dependency to stay backwards-compatible with existing
wiring. When it is absent the state gating / timeout extension steps are
skipped and a warning is logged — this is an acceptable fallback for
unit tests but production wiring must always inject both repositories.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.damage_repository import DamageRepository
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.value_objects.part_damage_compatibility import (
    is_compatible_pair,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
    RequestStatus,
)

_TIMEOUT_EXTENSION = timedelta(minutes=5)


def _ensure_mutable(request: RepairRequest) -> None:
    """Reject the mutation if the parent request is not in ``PRICING``."""
    if request.status is not RequestStatus.PRICING:
        raise ValueError(
            f"request {request.id} is in status {request.status.value}; "
            f"damage mutations require PRICING "
            f"({'terminal' if request.status in (RequestStatus.DONE, RequestStatus.FAILED) else 'pre-pricing'})."
        )


def _ensure_compatible(part_type: PartType, damage_type: DamageType) -> None:
    """Reject (part, damage) pairs that have no business meaning.

    The bot UI already filters incompatible buttons out of the keyboard;
    this is the defense-in-depth layer that catches stale callbacks from
    older keyboard versions, direct API clients and replayed payloads.
    """
    if not is_compatible_pair(part_type, damage_type):
        raise ValueError(
            f"damage_type={damage_type.value} is not compatible with "
            f"part_type={part_type.value}"
        )


async def _extend_timeout(
    requests: RepairRequestRepository, request: RepairRequest
) -> None:
    """Push ``timeout_at`` forward to keep the session alive for another window."""
    extended = request.with_extended_timeout(datetime.now(UTC) + _TIMEOUT_EXTENSION)
    await requests.update(extended)


@dataclass
class AddDamageInput:
    request_id: str
    part_type: PartType
    damage_type: DamageType
    source: DamageSource = DamageSource.MANUAL
    part_id: str | None = None
    confidence: float | None = None
    mask_image_key: str | None = None


@dataclass
class AddDamageResult:
    damage: DetectedDamage


class AddDamageUseCase:
    def __init__(self, request_repository: RepairRequestRepository, damage_repository: DamageRepository) -> None:
        self._requests = request_repository
        self._damages = damage_repository

    async def execute(self, data: AddDamageInput) -> AddDamageResult:
        request = await self._requests.get(data.request_id)
        if request is None:
            raise ValueError(f"request {data.request_id} not found")
        _ensure_mutable(request)
        # ML-sourced detections bypass compatibility gating: the detector is
        # constrained by its class vocabulary and a mismatch there is either
        # a model bug (logged and dropped in the worker) or a legitimate
        # edge case the human is about to correct via the edit flow. Manual
        # add, however, must not create nonsense.
        if data.source is DamageSource.MANUAL:
            _ensure_compatible(data.part_type, data.damage_type)

        damage = DetectedDamage(
            id=str(uuid4()),
            request_id=data.request_id,
            damage_type=data.damage_type,
            part_type=data.part_type,
            source=data.source,
            is_deleted=False,
            part_id=data.part_id,
            confidence=data.confidence,
            mask_image_key=data.mask_image_key,
        )
        await self._damages.add(damage)
        await _extend_timeout(self._requests, request)
        logger.info("Added damage id={} request_id={}", damage.id, damage.request_id)
        return AddDamageResult(damage=damage)


@dataclass
class EditDamageInput:
    """Input for editing a previously detected damage.

    ``damage_type`` is required; ``part_type`` is optional — when supplied,
    the damage is re-assigned to a different car part.

    ``request_id`` is optional for backwards compatibility but STRONGLY
    recommended in production: when present, the use case verifies that
    the damage actually belongs to that request (cross-request isolation).
    """

    damage_id: str
    damage_type: DamageType
    part_type: PartType | None = None
    request_id: str | None = None


@dataclass
class EditDamageResult:
    damage: DetectedDamage


class EditDamageUseCase:
    def __init__(
        self,
        damage_repository: DamageRepository,
        request_repository: RepairRequestRepository | None = None,
    ) -> None:
        self._damages = damage_repository
        self._requests = request_repository

    async def execute(self, data: EditDamageInput) -> EditDamageResult:
        damage = await self._damages.get(data.damage_id)
        if damage is None:
            raise ValueError(f"damage {data.damage_id} not found")
        if damage.is_deleted:
            raise ValueError(f"damage {data.damage_id} is deleted")

        if data.request_id is not None and damage.request_id != data.request_id:
            raise ValueError(
                f"damage {data.damage_id} does not belong to request {data.request_id}; "
                f"actual request_id={damage.request_id}"
            )

        if self._requests is not None:
            request = await self._requests.get(damage.request_id)
            if request is None:
                raise ValueError(f"parent request {damage.request_id} not found")
            _ensure_mutable(request)

        new_part_type = data.part_type if data.part_type is not None else damage.part_type
        # Edit always revalidates the final (part, damage) pair: the user
        # can change the type, the part, or both. Applies regardless of
        # whether the damage was originally ML-sourced — once a human edits
        # it, the pair must satisfy the same business rules as manual input.
        _ensure_compatible(new_part_type, data.damage_type)

        updated = DetectedDamage(
            id=damage.id,
            request_id=damage.request_id,
            damage_type=data.damage_type,
            part_type=new_part_type,
            source=damage.source,
            is_deleted=False,
            part_id=damage.part_id,
            confidence=damage.confidence,
            mask_image_key=damage.mask_image_key,
        )
        await self._damages.update(updated)

        if self._requests is not None:
            request = await self._requests.get(damage.request_id)
            if request is not None:
                await _extend_timeout(self._requests, request)

        logger.info(
            "Updated damage id={} new_type={} new_part={}",
            damage.id,
            data.damage_type,
            new_part_type,
        )
        return EditDamageResult(damage=updated)


@dataclass
class DeleteDamageInput:
    damage_id: str
    request_id: str | None = None


class DeleteDamageUseCase:
    def __init__(
        self,
        damage_repository: DamageRepository,
        request_repository: RepairRequestRepository | None = None,
    ) -> None:
        self._damages = damage_repository
        self._requests = request_repository

    async def execute(self, data: DeleteDamageInput) -> None:
        damage = await self._damages.get(data.damage_id)
        if damage is None:
            raise ValueError(f"damage {data.damage_id} not found")

        if data.request_id is not None and damage.request_id != data.request_id:
            raise ValueError(
                f"damage {data.damage_id} does not belong to request {data.request_id}; "
                f"actual request_id={damage.request_id}"
            )

        if self._requests is not None:
            request = await self._requests.get(damage.request_id)
            if request is None:
                raise ValueError(f"parent request {damage.request_id} not found")
            _ensure_mutable(request)

        await self._damages.soft_delete(data.damage_id)

        if self._requests is not None:
            request = await self._requests.get(damage.request_id)
            if request is not None:
                await _extend_timeout(self._requests, request)

        logger.info("Soft-deleted damage id={}", data.damage_id)
