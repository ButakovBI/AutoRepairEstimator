"""QA: editing a damage must preserve its provenance.

``DetectedDamage.source`` distinguishes ML-detected damages from manually
added ones. Downstream code and analytics rely on this: e.g. the ML
fallback path shows "Укажите повреждения вручную" only for MANUAL
sources, and the thesis reporting counts auto-detection accuracy based
on this field.

If ``EditDamageUseCase.execute`` ever accidentally resets ``source`` to
``MANUAL`` (say, because someone "cleaned up" the constructor call), every
ML-detected damage edited by the user would instantly look as if the user
entered it by hand, corrupting the metrics.

These tests pin the invariant:
  * ML damage stays ML after editing damage_type only,
  * ML damage stays ML after editing both damage_type and part_type,
  * Manual damage stays MANUAL.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.interfaces.damage_repository import DamageRepository
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
)
from auto_repair_estimator.backend.use_cases.manage_damages import (
    EditDamageInput,
    EditDamageUseCase,
)


class _InMemoryDamageRepo(DamageRepository):
    def __init__(self, damages: list[DetectedDamage]) -> None:
        self._damages: dict[str, DetectedDamage] = {d.id: d for d in damages}

    async def add(self, damage: DetectedDamage) -> None:
        self._damages[damage.id] = damage

    async def get(self, damage_id: str) -> DetectedDamage | None:
        return self._damages.get(damage_id)

    async def get_by_request_id(self, request_id: str) -> list[DetectedDamage]:
        return [d for d in self._damages.values() if d.request_id == request_id]

    async def update(self, damage: DetectedDamage) -> None:
        self._damages[damage.id] = damage

    async def soft_delete(self, damage_id: str) -> None:
        existing = self._damages[damage_id]
        self._damages[damage_id] = DetectedDamage(
            id=existing.id,
            request_id=existing.request_id,
            damage_type=existing.damage_type,
            part_type=existing.part_type,
            source=existing.source,
            is_deleted=True,
            part_id=existing.part_id,
            confidence=existing.confidence,
            mask_image_key=existing.mask_image_key,
        )


def _ml_damage_on_hood_with_scratch() -> DetectedDamage:
    # The ML model produced this record; confidence 0.85 ties the test's
    # intent to the "ML-detected" narrative — manual damages have no
    # confidence score set.
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-1",
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.ML,
        is_deleted=False,
        confidence=0.85,
    )


@pytest.mark.anyio
async def test_editing_damage_type_preserves_ml_source() -> None:
    damage = _ml_damage_on_hood_with_scratch()
    repo = _InMemoryDamageRepo([damage])
    use_case = EditDamageUseCase(damage_repository=repo)

    result = await use_case.execute(
        EditDamageInput(damage_id=damage.id, damage_type=DamageType.DENT)
    )

    assert result.damage.source is DamageSource.ML, (
        "EditDamageUseCase reset DamageSource.ML to MANUAL after a damage-type "
        "edit. This corrupts the 'auto-detected vs human-corrected' metric the "
        "thesis needs for its evaluation section."
    )


@pytest.mark.anyio
async def test_editing_part_type_preserves_ml_source() -> None:
    damage = _ml_damage_on_hood_with_scratch()
    repo = _InMemoryDamageRepo([damage])
    use_case = EditDamageUseCase(damage_repository=repo)

    result = await use_case.execute(
        EditDamageInput(
            damage_id=damage.id,
            damage_type=DamageType.DENT,
            part_type=PartType.DOOR,
        )
    )

    assert result.damage.source is DamageSource.ML
    # And confidence must survive too — it's the ML metadata that justifies
    # the ML label in the first place.
    assert result.damage.confidence == 0.85


@pytest.mark.anyio
async def test_editing_manual_damage_keeps_it_manual() -> None:
    """Symmetric check so we don't regress in the opposite direction."""
    manual = DetectedDamage(
        id=str(uuid4()),
        request_id="req-1",
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    repo = _InMemoryDamageRepo([manual])
    use_case = EditDamageUseCase(damage_repository=repo)

    result = await use_case.execute(
        EditDamageInput(damage_id=manual.id, damage_type=DamageType.DENT)
    )

    assert result.damage.source is DamageSource.MANUAL
