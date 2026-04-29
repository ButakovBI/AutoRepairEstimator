"""Defense-in-depth tests for part↔damage compatibility in the backend.

The bot UI is the first line of defense (it only renders compatible
buttons). These tests prove the backend is the second line: even if a
client bypasses the UI -- an old stale keyboard, a direct ``curl``, a
replayed Kafka payload -- the use case refuses to persist an incompatible
pair.
"""

from __future__ import annotations

import datetime as _dt
from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
    RequestMode,
    RequestStatus,
)
from auto_repair_estimator.backend.use_cases.manage_damages import (
    AddDamageInput,
    AddDamageUseCase,
    EditDamageInput,
    EditDamageUseCase,
)


class _FakeRequestRepo:
    def __init__(self) -> None:
        self._items: dict[str, RepairRequest] = {}

    async def add(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get(self, request_id: str) -> RepairRequest | None:
        return self._items.get(request_id)

    async def update(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get_timed_out_requests(self):  # type: ignore[no-untyped-def]
        return []


class _FakeDamageRepo:
    def __init__(self) -> None:
        self._items: dict[str, DetectedDamage] = {}

    async def add(self, damage: DetectedDamage) -> None:
        self._items[damage.id] = damage

    async def get(self, damage_id: str) -> DetectedDamage | None:
        return self._items.get(damage_id)

    async def get_by_request_id(self, request_id: str) -> list[DetectedDamage]:
        return [d for d in self._items.values() if d.request_id == request_id]

    async def update(self, damage: DetectedDamage) -> None:
        self._items[damage.id] = damage

    async def soft_delete(self, damage_id: str) -> None:  # pragma: no cover - not used here
        if damage_id in self._items:
            d = self._items[damage_id]
            self._items[damage_id] = DetectedDamage(
                id=d.id,
                request_id=d.request_id,
                damage_type=d.damage_type,
                part_type=d.part_type,
                source=d.source,
                is_deleted=True,
            )


def _make_pricing_request() -> RepairRequest:
    now = _dt.datetime.now(_dt.UTC)
    return RepairRequest(
        id=str(uuid4()),
        chat_id=1,
        user_id=2,
        mode=RequestMode.MANUAL,
        status=RequestStatus.PRICING,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "part,damage",
    [
        # These are the exact combos the user flagged as "should never be
        # offered" in the bug report: headlight/scratch, wheel/dent,
        # door/broken_glass.
        (PartType.HEADLIGHT, DamageType.SCRATCH),
        (PartType.HEADLIGHT, DamageType.CRACK),
        (PartType.WHEEL, DamageType.DENT),
        (PartType.WHEEL, DamageType.SCRATCH),
        (PartType.DOOR, DamageType.BROKEN_GLASS),
        (PartType.FRONT_WINDSHIELD, DamageType.SCRATCH),
    ],
)
async def test_add_damage_rejects_incompatible_pair(part: PartType, damage: DamageType) -> None:
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_pricing_request()
    await req_repo.add(req)
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=dmg_repo)

    with pytest.raises(ValueError, match="not compatible"):
        await use_case.execute(
            AddDamageInput(request_id=req.id, part_type=part, damage_type=damage)
        )

    # Must NOT persist a half-broken row.
    assert await dmg_repo.get_by_request_id(req.id) == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "part,damage",
    [
        (PartType.HOOD, DamageType.SCRATCH),
        (PartType.HEADLIGHT, DamageType.BROKEN_HEADLIGHT),
        (PartType.WHEEL, DamageType.FLAT_TIRE),
        (PartType.SIDE_WINDOW, DamageType.BROKEN_GLASS),
    ],
)
async def test_add_damage_accepts_compatible_pair(part: PartType, damage: DamageType) -> None:
    # Positive path: these are exactly the pairs the SSOT declares valid.
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_pricing_request()
    await req_repo.add(req)
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=dmg_repo)

    result = await use_case.execute(
        AddDamageInput(request_id=req.id, part_type=part, damage_type=damage)
    )

    assert result.damage.part_type is part
    assert result.damage.damage_type is damage


@pytest.mark.anyio
async def test_edit_damage_rejects_incompatible_final_pair() -> None:
    # The user edits a hood/scratch to become hood/broken_glass: the final
    # pair is incompatible with hood, so the edit must be rejected.
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_pricing_request()
    await req_repo.add(req)
    dmg = DetectedDamage(
        id=str(uuid4()),
        request_id=req.id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(dmg)
    use_case = EditDamageUseCase(damage_repository=dmg_repo, request_repository=req_repo)

    with pytest.raises(ValueError, match="not compatible"):
        await use_case.execute(
            EditDamageInput(damage_id=dmg.id, damage_type=DamageType.BROKEN_GLASS, request_id=req.id)
        )

    # The original damage must remain unchanged -- no partial write.
    fresh = await dmg_repo.get(dmg.id)
    assert fresh is not None
    assert fresh.damage_type is DamageType.SCRATCH


@pytest.mark.anyio
async def test_edit_damage_accepts_changing_part_to_keep_pair_compatible() -> None:
    # User originally picked door/dent (compatible) and then wants to move it
    # to headlight. The combined (headlight, dent) is NOT compatible, so the
    # edit must be rejected rather than silently stored.
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_pricing_request()
    await req_repo.add(req)
    dmg = DetectedDamage(
        id=str(uuid4()),
        request_id=req.id,
        damage_type=DamageType.DENT,
        part_type=PartType.DOOR,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(dmg)
    use_case = EditDamageUseCase(damage_repository=dmg_repo, request_repository=req_repo)

    with pytest.raises(ValueError, match="not compatible"):
        await use_case.execute(
            EditDamageInput(
                damage_id=dmg.id,
                damage_type=DamageType.DENT,
                part_type=PartType.HEADLIGHT,
                request_id=req.id,
            )
        )


@pytest.mark.anyio
async def test_add_damage_allows_ml_source_even_if_incompatible() -> None:
    # ML-sourced detections must bypass compatibility gating: the worker
    # already filters its output against DamageType enum, and any
    # semantically odd pair the ML outputs is the user's job to fix via the
    # edit flow. Blocking ingestion would lose data silently.
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_pricing_request()
    await req_repo.add(req)
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=dmg_repo)

    result = await use_case.execute(
        AddDamageInput(
            request_id=req.id,
            part_type=PartType.HEADLIGHT,
            damage_type=DamageType.SCRATCH,  # odd but ML sometimes hallucinates
            source=DamageSource.ML,
        )
    )

    assert result.damage.damage_type is DamageType.SCRATCH
    assert result.damage.source is DamageSource.ML
