from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.interfaces.damage_repository import DamageRepository
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageSource, DamageType, PartType


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
        logger.info("Added damage id={} request_id={}", damage.id, damage.request_id)
        return AddDamageResult(damage=damage)


@dataclass
class EditDamageInput:
    """Input for editing a previously detected damage.

    ``damage_type`` is required; ``part_type`` is optional — when supplied,
    the damage is re-assigned to a different car part, covering spec §3's
    "Изменить принадлежность повреждения к детали автомобиля" requirement.
    """

    damage_id: str
    damage_type: DamageType
    part_type: PartType | None = None


@dataclass
class EditDamageResult:
    damage: DetectedDamage


class EditDamageUseCase:
    def __init__(self, damage_repository: DamageRepository) -> None:
        self._damages = damage_repository

    async def execute(self, data: EditDamageInput) -> EditDamageResult:
        damage = await self._damages.get(data.damage_id)
        if damage is None:
            raise ValueError(f"damage {data.damage_id} not found")
        if damage.is_deleted:
            raise ValueError(f"damage {data.damage_id} is deleted")

        new_part_type = data.part_type if data.part_type is not None else damage.part_type

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


class DeleteDamageUseCase:
    def __init__(self, damage_repository: DamageRepository) -> None:
        self._damages = damage_repository

    async def execute(self, data: DeleteDamageInput) -> None:
        damage = await self._damages.get(data.damage_id)
        if damage is None:
            raise ValueError(f"damage {data.damage_id} not found")
        await self._damages.soft_delete(data.damage_id)
        logger.info("Soft-deleted damage id={}", data.damage_id)
