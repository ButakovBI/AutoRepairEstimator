"""Tests for the ``(request_id, part_type, damage_type)`` uniqueness invariant.

The invariant is enforced at three surfaces — each test exercises one:

* ``ProcessInferenceResultUseCase._save_damages`` — ML ingestion dedup.
* ``AddDamageUseCase`` — manual idempotency on duplicate adds.
* ``EditDamageUseCase._merge_on_conflict`` — merge on retype collision.

Tests go through the real use-cases (not the private helpers) so they
exercise the public contract and stay refactor-tolerant. We use the same
light fakes style as ``test_process_inference_result.py`` — no DB, no
outbox wiring beyond what each test reads back.
"""

from __future__ import annotations

import datetime as _dt
from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.detected_part import DetectedPart
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
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
from auto_repair_estimator.backend.use_cases.process_inference_result import (
    InferenceDamageData,
    InferencePartData,
    ProcessInferenceResultInput,
    ProcessInferenceResultUseCase,
)


# ---------------------------------------------------------------------------
# Minimal in-memory fakes — kept local to avoid coupling to other test files.
# ---------------------------------------------------------------------------


class _FakeRequestRepo:
    def __init__(self) -> None:
        self._items: dict[str, RepairRequest] = {}

    async def add(self, r: RepairRequest) -> None:
        self._items[r.id] = r

    async def get(self, rid: str) -> RepairRequest | None:
        return self._items.get(rid)

    async def update(self, r: RepairRequest) -> None:
        self._items[r.id] = r


class _FakePartRepo:
    def __init__(self) -> None:
        self._items: list[DetectedPart] = []

    async def add(self, p: DetectedPart) -> None:
        self._items.append(p)

    async def get_by_request_id(self, rid: str) -> list[DetectedPart]:
        return [p for p in self._items if p.request_id == rid]


class _FakeDamageRepo:
    def __init__(self) -> None:
        self._items: list[DetectedDamage] = []

    async def add(self, d: DetectedDamage) -> None:
        self._items.append(d)

    async def get_by_request_id(self, rid: str) -> list[DetectedDamage]:
        return [d for d in self._items if d.request_id == rid]

    async def get(self, did: str) -> DetectedDamage | None:
        return next((d for d in self._items if d.id == did), None)

    async def update(self, d: DetectedDamage) -> None:
        for i, item in enumerate(self._items):
            if item.id == d.id:
                self._items[i] = d
                return

    async def soft_delete(self, did: str) -> None:
        for i, item in enumerate(self._items):
            if item.id == did:
                self._items[i] = DetectedDamage(
                    id=item.id,
                    request_id=item.request_id,
                    damage_type=item.damage_type,
                    part_type=item.part_type,
                    source=item.source,
                    is_deleted=True,
                    part_id=item.part_id,
                    confidence=item.confidence,
                    mask_image_key=item.mask_image_key,
                )
                return


class _FakeOutboxRepo:
    def __init__(self) -> None:
        self._items: list[OutboxEvent] = []

    async def add(self, e: OutboxEvent) -> None:
        self._items.append(e)

    async def get_unpublished(self, limit: int) -> list[OutboxEvent]:
        return [e for e in self._items if e.published_at is None][:limit]

    async def mark_published(self, ids: list[str]) -> None:
        pass


def _make_request(
    req_id: str, status: RequestStatus = RequestStatus.PRICING
) -> RepairRequest:
    now = _dt.datetime.now(_dt.UTC)
    return RepairRequest(
        id=req_id,
        chat_id=123,
        user_id=456,
        mode=RequestMode.ML,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )


# ---------------------------------------------------------------------------
# ML ingestion
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ml_ingestion_collapses_duplicate_part_damage_pairs() -> None:
    """17 bumper scratches from the detector must land as 1 stored damage.

    Mirrors the real failure mode reported by the user (26 bumper +
    scratch rows from a single photo). Keeps ordering across distinct
    pairs, so a `(bumper, crack)` detection isn't swallowed by earlier
    `(bumper, scratch)` duplicates.
    """

    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id, RequestStatus.PROCESSING))
    use_case = ProcessInferenceResultUseCase(
        request_repository=req_repo,
        part_repository=_FakePartRepo(),
        damage_repository=dmg_repo,
        outbox_repository=_FakeOutboxRepo(),
        state_machine=RequestStateMachine(),
        notifications_topic="notifications",
    )

    dup_scratches = [
        InferenceDamageData("scratch", "bumper", 0.9, None) for _ in range(17)
    ]
    distinct_crack = InferenceDamageData("crack", "bumper", 0.7, None)

    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[InferencePartData("bumper", 0.9, [0, 0, 1, 1], None)],
            damages=[*dup_scratches, distinct_crack],
            composited_image_key=None,
            error_message=None,
        )
    )

    stored = await dmg_repo.get_by_request_id(req_id)
    pairs = {(d.part_type, d.damage_type) for d in stored}
    assert pairs == {
        (PartType.BUMPER, DamageType.SCRATCH),
        (PartType.BUMPER, DamageType.CRACK),
    }


@pytest.mark.anyio
async def test_ml_ingestion_notification_uses_deduped_damage_list() -> None:
    """Notification payload echoes what was persisted, not the raw detector output.

    The bot renders the "Обнаруженные повреждения" card from this
    payload; if we echoed the raw list the user would see 17 scratches
    while the edit screen shows 1.
    """

    req_repo = _FakeRequestRepo()
    outbox_repo = _FakeOutboxRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id, RequestStatus.PROCESSING))
    use_case = ProcessInferenceResultUseCase(
        request_repository=req_repo,
        part_repository=_FakePartRepo(),
        damage_repository=_FakeDamageRepo(),
        outbox_repository=outbox_repo,
        state_machine=RequestStateMachine(),
        notifications_topic="notifications",
    )

    await use_case.execute(
        ProcessInferenceResultInput(
            request_id=req_id,
            status="success",
            parts=[InferencePartData("door", 0.9, [0, 0, 1, 1], None)],
            damages=[
                InferenceDamageData("scratch", "door", 0.9, None),
                InferenceDamageData("scratch", "door", 0.8, None),
                InferenceDamageData("scratch", "door", 0.7, None),
            ],
            composited_image_key=None,
            error_message=None,
        )
    )

    events = await outbox_repo.get_unpublished(10)
    assert len(events) == 1
    damages = events[0].payload["damages"]
    # One dedup pair → one entry in the payload.
    assert damages == [{"part_type": "door", "damage_type": "scratch"}]


# ---------------------------------------------------------------------------
# Manual add
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_manual_add_is_idempotent_on_duplicate_pair() -> None:
    """Adding ``(door, scratch)`` twice returns the same row, not a new one."""

    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id))
    use_case = AddDamageUseCase(req_repo, dmg_repo)

    first = await use_case.execute(
        AddDamageInput(
            request_id=req_id,
            part_type=PartType.DOOR,
            damage_type=DamageType.SCRATCH,
        )
    )
    second = await use_case.execute(
        AddDamageInput(
            request_id=req_id,
            part_type=PartType.DOOR,
            damage_type=DamageType.SCRATCH,
        )
    )

    assert first.damage.id == second.damage.id
    assert second.already_existed is True
    assert first.already_existed is False
    stored = await dmg_repo.get_by_request_id(req_id)
    assert len([d for d in stored if not d.is_deleted]) == 1


@pytest.mark.anyio
async def test_manual_add_after_delete_creates_fresh_row() -> None:
    """Soft-deleted duplicates must not resurrect on re-add.

    A deleted damage should stay deleted — the user expects a new,
    editable entry. Otherwise the audit trail (who deleted / when) gets
    silently overwritten on the next click.
    """

    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id))
    use_case = AddDamageUseCase(req_repo, dmg_repo)

    first = await use_case.execute(
        AddDamageInput(
            request_id=req_id,
            part_type=PartType.DOOR,
            damage_type=DamageType.SCRATCH,
        )
    )
    await dmg_repo.soft_delete(first.damage.id)

    second = await use_case.execute(
        AddDamageInput(
            request_id=req_id,
            part_type=PartType.DOOR,
            damage_type=DamageType.SCRATCH,
        )
    )

    assert second.already_existed is False
    assert second.damage.id != first.damage.id


# ---------------------------------------------------------------------------
# Edit retype merge
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_edit_merges_into_conflicting_sibling() -> None:
    """Retyping A=(door, scratch) to (door, crack) when B=(door, crack) exists.

    The edited damage wins (latest user intent); the sibling is soft-
    deleted. Exactly one active damage with the target pair must remain
    on the request afterwards.
    """

    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id))

    # Seed two damages on the same part, different types.
    scratch = DetectedDamage(
        id=str(uuid4()),
        request_id=req_id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.DOOR,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    crack = DetectedDamage(
        id=str(uuid4()),
        request_id=req_id,
        damage_type=DamageType.CRACK,
        part_type=PartType.DOOR,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(scratch)
    await dmg_repo.add(crack)

    use_case = EditDamageUseCase(dmg_repo, req_repo)

    # Retype the scratch → crack. Collides with the existing crack.
    await use_case.execute(
        EditDamageInput(
            damage_id=scratch.id,
            damage_type=DamageType.CRACK,
            request_id=req_id,
        )
    )

    stored = await dmg_repo.get_by_request_id(req_id)
    active = [d for d in stored if not d.is_deleted]
    assert len(active) == 1
    # The edited damage (scratch's id) is the one that survives.
    assert active[0].id == scratch.id
    assert active[0].damage_type is DamageType.CRACK


@pytest.mark.anyio
async def test_edit_without_conflict_leaves_others_untouched() -> None:
    """Retyping to a non-colliding pair must not touch any other damage."""

    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req_id = str(uuid4())
    await req_repo.add(_make_request(req_id))

    a = DetectedDamage(
        id=str(uuid4()),
        request_id=req_id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.DOOR,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    b = DetectedDamage(
        id=str(uuid4()),
        request_id=req_id,
        damage_type=DamageType.DENT,
        part_type=PartType.BUMPER,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(a)
    await dmg_repo.add(b)

    use_case = EditDamageUseCase(dmg_repo, req_repo)
    # Change only a's type; no collision with b.
    await use_case.execute(
        EditDamageInput(
            damage_id=a.id,
            damage_type=DamageType.RUST,
            request_id=req_id,
        )
    )

    stored = await dmg_repo.get_by_request_id(req_id)
    assert {d.id for d in stored if not d.is_deleted} == {a.id, b.id}
