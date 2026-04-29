"""Behavioral integration tests for PostgresDamageRepository.

Each test verifies one observable behaviour of the damage repository
against a real PostgreSQL database.  Uses AAA structure throughout.
"""

from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

from auto_repair_estimator.backend.adapters.repositories.postgres_damage_repository import PostgresDamageRepository
from auto_repair_estimator.backend.adapters.repositories.postgres_repair_request_repository import (
    PostgresRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
    RequestMode,
)


async def _seed_request(pool: asyncpg.Pool) -> str:
    """Insert a PRICING-mode repair request and return its string UUID."""
    request = RepairRequest.new(
        request_id=str(uuid4()),
        chat_id=1,
        user_id=None,
        mode=RequestMode.MANUAL,
    )
    await PostgresRepairRequestRepository(pool).add(request)
    return request.id


def _make_damage(request_id: str, damage_type: DamageType = DamageType.SCRATCH) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id=request_id,
        part_type=PartType.HOOD,
        damage_type=damage_type,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )


@pytest.mark.anyio
async def test_added_damage_is_retrievable_by_request_id(db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _seed_request(db_pool)
    repo = PostgresDamageRepository(db_pool)
    damage = _make_damage(request_id)

    # Act
    await repo.add(damage)
    results = await repo.get_by_request_id(request_id)

    # Assert
    assert len(results) == 1
    assert results[0].id == damage.id
    assert results[0].part_type is PartType.HOOD
    assert results[0].damage_type is DamageType.SCRATCH
    assert results[0].source is DamageSource.MANUAL
    assert not results[0].is_deleted


@pytest.mark.anyio
async def test_get_by_request_id_returns_empty_list_when_no_damages(db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _seed_request(db_pool)
    repo = PostgresDamageRepository(db_pool)

    # Act
    results = await repo.get_by_request_id(request_id)

    # Assert
    assert results == []


@pytest.mark.anyio
async def test_get_returns_none_for_unknown_damage_id(db_pool: asyncpg.Pool) -> None:
    # Arrange
    repo = PostgresDamageRepository(db_pool)

    # Act
    result = await repo.get(str(uuid4()))

    # Assert
    assert result is None


@pytest.mark.anyio
async def test_damage_type_change_is_persisted_after_update(db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _seed_request(db_pool)
    repo = PostgresDamageRepository(db_pool)
    damage = _make_damage(request_id, DamageType.SCRATCH)
    await repo.add(damage)

    # Act
    damage.damage_type = DamageType.DENT
    await repo.update(damage)
    refreshed = await repo.get(damage.id)

    # Assert
    assert refreshed is not None
    assert refreshed.damage_type is DamageType.DENT


@pytest.mark.anyio
async def test_soft_delete_marks_damage_as_deleted_without_removing_the_row(db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _seed_request(db_pool)
    repo = PostgresDamageRepository(db_pool)
    damage = _make_damage(request_id)
    await repo.add(damage)

    # Act
    await repo.soft_delete(damage.id)
    refreshed = await repo.get(damage.id)

    # Assert — row still exists but is flagged as deleted
    assert refreshed is not None
    assert refreshed.is_deleted is True


@pytest.mark.anyio
async def test_soft_deleted_damage_remains_visible_in_get_by_request_id(db_pool: asyncpg.Pool) -> None:
    """Callers receive soft-deleted damages and decide how to handle them."""
    # Arrange
    request_id = await _seed_request(db_pool)
    repo = PostgresDamageRepository(db_pool)
    damage = _make_damage(request_id)
    await repo.add(damage)
    await repo.soft_delete(damage.id)

    # Act
    all_damages = await repo.get_by_request_id(request_id)

    # Assert
    assert len(all_damages) == 1
    assert all_damages[0].is_deleted is True


@pytest.mark.anyio
async def test_all_damages_for_a_request_are_returned(db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _seed_request(db_pool)
    repo = PostgresDamageRepository(db_pool)
    for dtype in (DamageType.SCRATCH, DamageType.DENT, DamageType.RUST):
        await repo.add(_make_damage(request_id, dtype))

    # Act
    results = await repo.get_by_request_id(request_id)

    # Assert
    assert len(results) == 3
    found_types = {r.damage_type for r in results}
    assert found_types == {DamageType.SCRATCH, DamageType.DENT, DamageType.RUST}
