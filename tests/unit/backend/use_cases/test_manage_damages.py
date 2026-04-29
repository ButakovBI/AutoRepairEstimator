"""Unit tests for the damage management use cases.

Each test covers exactly one observable behaviour of AddDamageUseCase,
EditDamageUseCase, or DeleteDamageUseCase.  All infrastructure is replaced
with simple in-memory fakes so the tests stay fast and aren't coupled to
database details.  AAA structure is explicit throughout.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
    RequestMode,
)
from auto_repair_estimator.backend.use_cases.manage_damages import (
    AddDamageInput,
    AddDamageUseCase,
    DeleteDamageInput,
    DeleteDamageUseCase,
    EditDamageInput,
    EditDamageUseCase,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRequestRepo:
    def __init__(self) -> None:
        self._items: dict[str, RepairRequest] = {}

    async def add(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get(self, request_id: str) -> RepairRequest | None:
        return self._items.get(request_id)

    async def update(self, request: RepairRequest) -> None:
        self._items[request.id] = request


class _FakeDamageRepo:
    def __init__(self) -> None:
        self._items: dict[str, DetectedDamage] = {}

    async def add(self, damage: DetectedDamage) -> None:
        self._items[damage.id] = damage

    async def get_by_request_id(self, request_id: str) -> list[DetectedDamage]:
        return [d for d in self._items.values() if d.request_id == request_id]

    async def get(self, damage_id: str) -> DetectedDamage | None:
        return self._items.get(damage_id)

    async def update(self, damage: DetectedDamage) -> None:
        self._items[damage.id] = damage

    async def soft_delete(self, damage_id: str) -> None:
        if damage_id in self._items:
            d = self._items[damage_id]
            self._items[damage_id] = DetectedDamage(
                id=d.id,
                request_id=d.request_id,
                damage_type=d.damage_type,
                part_type=d.part_type,
                source=d.source,
                is_deleted=True,
                part_id=d.part_id,
                confidence=d.confidence,
                mask_image_key=d.mask_image_key,
            )


def _make_pricing_request(request_id: str) -> RepairRequest:
    return RepairRequest.new(
        request_id=request_id,
        chat_id=123,
        user_id=456,
        mode=RequestMode.MANUAL,
    )


# ---------------------------------------------------------------------------
# AddDamageUseCase
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_add_damage_stores_new_damage_with_manual_source() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    request_id = str(uuid4())
    await req_repo.add(_make_pricing_request(request_id))
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=dmg_repo)

    # Act
    result = await use_case.execute(
        AddDamageInput(request_id=request_id, part_type=PartType.HOOD, damage_type=DamageType.SCRATCH)
    )

    # Assert — returned damage matches input and is tagged as manual
    assert result.damage.part_type is PartType.HOOD
    assert result.damage.damage_type is DamageType.SCRATCH
    assert result.damage.source is DamageSource.MANUAL
    assert not result.damage.is_deleted


@pytest.mark.anyio
async def test_add_damage_persists_exactly_one_record_in_repository() -> None:
    # Arrange
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    request_id = str(uuid4())
    await req_repo.add(_make_pricing_request(request_id))
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=dmg_repo)

    # Act
    await use_case.execute(
        AddDamageInput(request_id=request_id, part_type=PartType.HOOD, damage_type=DamageType.SCRATCH)
    )

    # Assert — repository has exactly the one new damage
    damages = await dmg_repo.get_by_request_id(request_id)
    assert len(damages) == 1


@pytest.mark.anyio
async def test_add_damage_raises_value_error_for_nonexistent_request() -> None:
    # Arrange
    use_case = AddDamageUseCase(request_repository=_FakeRequestRepo(), damage_repository=_FakeDamageRepo())

    # Act / Assert
    with pytest.raises(ValueError, match="not found"):
        await use_case.execute(
            AddDamageInput(request_id="nonexistent", part_type=PartType.HOOD, damage_type=DamageType.SCRATCH)
        )


# ---------------------------------------------------------------------------
# EditDamageUseCase
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_edit_damage_changes_damage_type_in_repository() -> None:
    # Arrange
    dmg_repo = _FakeDamageRepo()
    damage_id = str(uuid4())
    original = DetectedDamage(
        id=damage_id,
        request_id="req-1",
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.ML,
        is_deleted=False,
    )
    await dmg_repo.add(original)
    use_case = EditDamageUseCase(damage_repository=dmg_repo)

    # Act
    result = await use_case.execute(EditDamageInput(damage_id=damage_id, damage_type=DamageType.DENT))

    # Assert — result and stored record both reflect the new type
    assert result.damage.damage_type is DamageType.DENT
    stored = await dmg_repo.get(damage_id)
    assert stored is not None
    assert stored.damage_type is DamageType.DENT


@pytest.mark.anyio
async def test_edit_damage_preserves_part_type_and_source() -> None:
    # Arrange
    dmg_repo = _FakeDamageRepo()
    damage_id = str(uuid4())
    original = DetectedDamage(
        id=damage_id,
        request_id="req-1",
        damage_type=DamageType.SCRATCH,
        part_type=PartType.BUMPER,
        source=DamageSource.ML,
        is_deleted=False,
    )
    await dmg_repo.add(original)
    use_case = EditDamageUseCase(damage_repository=dmg_repo)

    # Act
    result = await use_case.execute(EditDamageInput(damage_id=damage_id, damage_type=DamageType.RUST))

    # Assert — side fields unchanged
    assert result.damage.part_type is PartType.BUMPER
    assert result.damage.source is DamageSource.ML


@pytest.mark.anyio
async def test_edit_damage_raises_value_error_for_soft_deleted_damage() -> None:
    # Arrange
    dmg_repo = _FakeDamageRepo()
    damage_id = str(uuid4())
    deleted = DetectedDamage(
        id=damage_id,
        request_id="req-1",
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.ML,
        is_deleted=True,
    )
    await dmg_repo.add(deleted)
    use_case = EditDamageUseCase(damage_repository=dmg_repo)

    # Act / Assert
    with pytest.raises(ValueError, match="deleted"):
        await use_case.execute(EditDamageInput(damage_id=damage_id, damage_type=DamageType.DENT))


# ---------------------------------------------------------------------------
# DeleteDamageUseCase
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_damage_marks_existing_damage_as_deleted() -> None:
    # Arrange
    dmg_repo = _FakeDamageRepo()
    damage_id = str(uuid4())
    damage = DetectedDamage(
        id=damage_id,
        request_id="req-1",
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(damage)
    use_case = DeleteDamageUseCase(damage_repository=dmg_repo)

    # Act
    await use_case.execute(DeleteDamageInput(damage_id=damage_id))

    # Assert — record is soft-deleted, not removed
    stored = await dmg_repo.get(damage_id)
    assert stored is not None
    assert stored.is_deleted is True


@pytest.mark.anyio
async def test_delete_damage_raises_value_error_for_nonexistent_damage() -> None:
    # Arrange
    use_case = DeleteDamageUseCase(damage_repository=_FakeDamageRepo())

    # Act / Assert
    with pytest.raises(ValueError, match="not found"):
        await use_case.execute(DeleteDamageInput(damage_id="nonexistent"))
