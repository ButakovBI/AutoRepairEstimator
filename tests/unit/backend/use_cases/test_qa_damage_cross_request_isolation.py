"""QA: damages of one request must be isolated from other requests.

The REST API exposes ``PATCH /v1/requests/{request_id}/damages/{damage_id}``
and ``DELETE /v1/requests/{request_id}/damages/{damage_id}``. A sane contract
is:

* ``damage_id`` is looked up only inside ``request_id``;
* mismatching IDs produce 404 (or ValueError at the use case layer).

This keeps each chat session's damages private and prevents a buggy/abusive
client from editing another session's basket simply by knowing a damage UUID.

The implementation today takes only ``damage_id`` at the use-case layer and
ignores ``request_id`` at the API layer, so the isolation is not enforced.
The tests below exercise exactly that boundary.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
)
from auto_repair_estimator.backend.use_cases.manage_damages import (
    DeleteDamageInput,
    DeleteDamageUseCase,
    EditDamageInput,
    EditDamageUseCase,
)


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
            )


def _make_damage(request_id: str) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id=request_id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )


# ---------------------------------------------------------------------------
# EditDamageInput carries request_id
# ---------------------------------------------------------------------------


def test_edit_damage_input_includes_request_id_field() -> None:
    """Without a ``request_id`` in the input, the use case has no way to
    verify that the damage being edited actually belongs to the URL's
    request — opening the door to cross-session tampering."""

    import dataclasses

    field_names = {f.name for f in dataclasses.fields(EditDamageInput)}
    assert "request_id" in field_names, (
        "EditDamageInput lacks a 'request_id' field. "
        "PATCH /v1/requests/{rid}/damages/{did} can therefore modify a damage "
        "that does NOT belong to {rid}. Add 'request_id' to the input and "
        "check it against damage.request_id before updating."
    )


def test_delete_damage_input_includes_request_id_field() -> None:
    """Same contract as for editing: deletion must be scoped to one request."""

    import dataclasses

    field_names = {f.name for f in dataclasses.fields(DeleteDamageInput)}
    assert "request_id" in field_names, (
        "DeleteDamageInput lacks a 'request_id' field — DELETE on one "
        "request can soft-delete a damage attached to a completely different "
        "request as long as the caller knows the damage_id."
    )


# ---------------------------------------------------------------------------
# Behavioural: edit/delete across requests must fail
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_edit_damage_rejects_damage_belonging_to_another_request() -> None:
    """Given a damage that belongs to ``request A``, editing it with the
    URL context of ``request B`` must raise ValueError (not silently
    succeed)."""

    # Arrange
    dmg_repo = _FakeDamageRepo()
    damage_of_request_a = _make_damage(request_id="request-A")
    await dmg_repo.add(damage_of_request_a)
    use_case = EditDamageUseCase(damage_repository=dmg_repo)

    # Act / Assert — we pretend the caller says "this damage lives in request B".
    # Until the bug is fixed, EditDamageInput won't accept ``request_id`` as a
    # kwarg, so we try/except to surface the exact structural issue.
    try:
        await use_case.execute(
            EditDamageInput(
                damage_id=damage_of_request_a.id,
                damage_type=DamageType.DENT,
                request_id="request-B",  # type: ignore[call-arg]
            )
        )
    except TypeError as exc:
        pytest.fail(
            "EditDamageInput does not accept a 'request_id' kwarg, so the use case "
            f"cannot enforce cross-request isolation. Underlying error: {exc}"
        )
    except ValueError as exc:
        # This is the happy path — ValueError because of wrong request_id.
        msg = str(exc).lower()
        assert "request" in msg, (
            f"Use case raised ValueError but the message {msg!r} does not mention "
            "the mismatched request_id. Downstream error logs will be ambiguous."
        )
        return

    # If no exception, the bug is present: the damage was edited cross-request.
    stored = await dmg_repo.get(damage_of_request_a.id)
    assert stored is None or stored.damage_type is DamageType.SCRATCH, (
        "EditDamageUseCase silently edited a damage that belongs to request-A "
        "while the caller claimed request-B; cross-request isolation is broken."
    )
    pytest.fail(
        "EditDamageUseCase did not raise on a cross-request mutation attempt — "
        "any client that knows a damage UUID can rewrite another session's basket."
    )


@pytest.mark.anyio
async def test_delete_damage_rejects_damage_belonging_to_another_request() -> None:
    """Same contract as above, for soft-delete."""

    # Arrange
    dmg_repo = _FakeDamageRepo()
    damage_of_request_a = _make_damage(request_id="request-A")
    await dmg_repo.add(damage_of_request_a)
    use_case = DeleteDamageUseCase(damage_repository=dmg_repo)

    # Act / Assert
    try:
        await use_case.execute(
            DeleteDamageInput(
                damage_id=damage_of_request_a.id,
                request_id="request-B",  # type: ignore[call-arg]
            )
        )
    except TypeError as exc:
        pytest.fail(
            "DeleteDamageInput does not accept a 'request_id' kwarg, so the use case "
            f"cannot enforce cross-request isolation. Underlying error: {exc}"
        )
    except ValueError as exc:
        assert "request" in str(exc).lower(), (
            "DeleteDamageUseCase raised ValueError but the message does not refer to "
            "the cross-request mismatch."
        )
        return

    stored = await dmg_repo.get(damage_of_request_a.id)
    assert stored is not None and stored.is_deleted is False, (
        "DeleteDamageUseCase soft-deleted damage_of_request_a from the context of "
        "request-B; cross-request isolation is broken."
    )
    pytest.fail(
        "DeleteDamageUseCase did not raise on a cross-request delete attempt — "
        "a client knowing a damage UUID can delete another session's damages."
    )
