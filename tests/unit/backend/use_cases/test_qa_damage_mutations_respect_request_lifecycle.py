"""QA: damage mutations must respect the parent repair-request lifecycle.

Per the architectural plan, a ``RepairRequest`` goes through the state machine

    CREATED -> QUEUED -> PROCESSING -> PRICING -> DONE
                                               \\-> FAILED

``DONE`` and ``FAILED`` are terminal. A request in a terminal state represents
a closed business transaction: the user has already seen the final estimate
(``DONE``) or has been notified of the failure (``FAILED``). Mutating the
damage list of such a request would:

* retroactively change an estimate the user already received, or
* silently "resurrect" a failed session without re-running the state machine.

Both are business-logic violations, so the Add/Edit/Delete damage use cases
MUST refuse to mutate damages that belong to a request in a terminal state.

The tests below fail today because none of the three use cases checks the
parent request's status — ``AddDamageUseCase`` only checks that the request
exists, and ``EditDamageUseCase`` / ``DeleteDamageUseCase`` don't even look
at the parent request.
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
    DeleteDamageInput,
    DeleteDamageUseCase,
    EditDamageInput,
    EditDamageUseCase,
)


# ---------------------------------------------------------------------------
# Minimal test doubles — no infrastructure.
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

    async def get(self, damage_id: str) -> DetectedDamage | None:
        return self._items.get(damage_id)

    async def get_by_request_id(self, request_id: str) -> list[DetectedDamage]:
        return [d for d in self._items.values() if d.request_id == request_id]

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


def _request_with_status(status: RequestStatus) -> RepairRequest:
    now = _dt.datetime.now(_dt.UTC)
    return RepairRequest(
        id=str(uuid4()),
        chat_id=42,
        user_id=7,
        mode=RequestMode.MANUAL,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )


def _seed_damage(repo: _FakeDamageRepo, request_id: str) -> DetectedDamage:
    damage = DetectedDamage(
        id=str(uuid4()),
        request_id=request_id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    # Synchronous shim — the fake's add doesn't need awaiting here.
    repo._items[damage.id] = damage  # noqa: SLF001
    return damage


# The terminal statuses we expect to reject damage mutations on.
_TERMINAL_STATUSES = [RequestStatus.DONE, RequestStatus.FAILED]


# ---------------------------------------------------------------------------
# AddDamageUseCase
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("status", _TERMINAL_STATUSES, ids=lambda s: s.value)
async def test_add_damage_rejects_when_parent_request_is_terminal(status: RequestStatus) -> None:
    """Adding a new damage to a DONE/FAILED request must raise ``ValueError``.

    If it doesn't, the user's final estimate can silently change after they
    have already been shown a total cost and (for DONE) confirmed it.
    """

    # Arrange
    req_repo = _FakeRequestRepo()
    terminal_request = _request_with_status(status)
    await req_repo.add(terminal_request)
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=_FakeDamageRepo())

    # Act / Assert
    with pytest.raises(ValueError) as exc_info:
        await use_case.execute(
            AddDamageInput(
                request_id=terminal_request.id,
                part_type=PartType.HOOD,
                damage_type=DamageType.DENT,
            )
        )
    # The message should mention the status so ops can diagnose it.
    assert status.value in str(exc_info.value).lower() or "terminal" in str(exc_info.value).lower(), (
        f"AddDamageUseCase accepted a mutation on a {status.value} request. "
        "Error message was: " + repr(str(exc_info.value))
    )


# ---------------------------------------------------------------------------
# EditDamageUseCase
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("status", _TERMINAL_STATUSES, ids=lambda s: s.value)
async def test_edit_damage_rejects_when_parent_request_is_terminal(status: RequestStatus) -> None:
    """Editing a damage on a DONE/FAILED request must raise ``ValueError``.

    Otherwise any client who knows a ``damage_id`` can rewrite the priced
    basket of a closed session.
    """

    # Arrange
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    terminal_request = _request_with_status(status)
    await req_repo.add(terminal_request)
    damage = _seed_damage(dmg_repo, terminal_request.id)

    # EditDamageUseCase as currently wired only sees a damage repo; a fix
    # needs to either inject the request repo or carry status on the damage
    # itself. Either way, the contract we want to verify is behavioural:
    # editing must fail. We therefore pass both repos through a tiny
    # subclass that the eventual fix will satisfy.
    use_case = EditDamageUseCase(damage_repository=dmg_repo)
    # Attach the request repo through the same attribute name we expect the
    # fix to use, so the test documents the intended dependency direction.
    use_case._requests = req_repo  # type: ignore[attr-defined]  # noqa: SLF001

    # Act / Assert
    with pytest.raises(ValueError) as exc_info:
        await use_case.execute(
            EditDamageInput(
                damage_id=damage.id,
                damage_type=DamageType.DENT,
            )
        )
    msg = str(exc_info.value).lower()
    assert status.value in msg or "terminal" in msg or "closed" in msg, (
        f"EditDamageUseCase mutated damage {damage.id} on a {status.value} request. "
        "Error message was: " + repr(str(exc_info.value))
    )


# ---------------------------------------------------------------------------
# DeleteDamageUseCase
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("status", _TERMINAL_STATUSES, ids=lambda s: s.value)
async def test_delete_damage_rejects_when_parent_request_is_terminal(status: RequestStatus) -> None:
    """Soft-deleting a damage on a DONE/FAILED request must raise ``ValueError``.

    Allowing deletion would shrink the estimate of a closed transaction
    retroactively.
    """

    # Arrange
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    terminal_request = _request_with_status(status)
    await req_repo.add(terminal_request)
    damage = _seed_damage(dmg_repo, terminal_request.id)

    use_case = DeleteDamageUseCase(damage_repository=dmg_repo)
    use_case._requests = req_repo  # type: ignore[attr-defined]  # noqa: SLF001

    # Act / Assert
    with pytest.raises(ValueError) as exc_info:
        await use_case.execute(DeleteDamageInput(damage_id=damage.id))
    msg = str(exc_info.value).lower()
    assert status.value in msg or "terminal" in msg or "closed" in msg, (
        f"DeleteDamageUseCase soft-deleted damage {damage.id} on a {status.value} request. "
        "Error message was: " + repr(str(exc_info.value))
    )
