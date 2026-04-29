"""QA spec-compliance tests for damage editing.

Requirements §3 (functional description of the bot's edit flow) lists four
edit operations the user must be able to perform after ML inference:

    - Изменить тип повреждения
    - Изменить принадлежность повреждения к детали автомобиля
    - Удалить ошибочно найденное повреждение
    - Добавить новое необнаруженное повреждение

The codebase ships ``AddDamageUseCase``, ``DeleteDamageUseCase``, and
``EditDamageUseCase``. However, ``EditDamageUseCase`` currently only allows
changing the ``damage_type`` field — it does NOT expose a way to re-assign a
damage to a different car part. That violates the second bullet above.

These tests codify the missing capability. They intentionally use public
contracts only (dataclass fields and the ``execute`` call signature) so the
tests stay meaningful under refactoring.
"""

from __future__ import annotations

import dataclasses
from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
)
from auto_repair_estimator.backend.use_cases.manage_damages import (
    EditDamageInput,
    EditDamageUseCase,
)


class _FakeDamageRepo:
    """Minimal in-memory damage repo satisfying the DamageRepository protocol."""

    def __init__(self) -> None:
        self._items: dict[str, DetectedDamage] = {}

    async def add(self, d: DetectedDamage) -> None:
        self._items[d.id] = d

    async def get(self, did: str) -> DetectedDamage | None:
        return self._items.get(did)

    async def update(self, d: DetectedDamage) -> None:
        self._items[d.id] = d

    async def soft_delete(self, did: str) -> None:
        self._items[did] = dataclasses.replace(self._items[did], is_deleted=True)

    async def get_by_request_id(self, rid: str) -> list[DetectedDamage]:
        return [d for d in self._items.values() if d.request_id == rid]


def _make_damage(part: PartType, damage: DamageType) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-1",
        damage_type=damage,
        part_type=part,
        source=DamageSource.ML,
        is_deleted=False,
    )


def test_edit_damage_input_exposes_part_type_field() -> None:
    """EditDamageInput must let callers specify a new part_type.

    Without this field on the public input dataclass there is no way for the
    bot handler or the HTTP API to ask the use case to re-assign a damage to
    a different part, which makes the 2nd bullet of requirement §3 impossible
    to satisfy through the documented architecture.
    """
    field_names = {f.name for f in dataclasses.fields(EditDamageInput)}
    assert "part_type" in field_names, (
        "EditDamageInput is missing the 'part_type' field — users cannot change "
        "damage-to-part assignment, which is required by the spec §3 bullet "
        "'Изменить принадлежность повреждения к детали автомобиля'. "
        f"Current fields: {sorted(field_names)}"
    )


@pytest.mark.anyio
async def test_edit_damage_use_case_reassigns_damage_to_a_different_part() -> None:
    """Executing the edit flow must be able to move a damage between parts.

    Scenario: ML mis-classified a scratch as belonging to the hood while it
    actually belongs to the bumper. The user fixes the mistake via the bot.
    After edit, the persisted damage must reflect the new part_type.
    """
    repo = _FakeDamageRepo()
    damage = _make_damage(PartType.HOOD, DamageType.SCRATCH)
    await repo.add(damage)
    use_case = EditDamageUseCase(damage_repository=repo)

    # Construct the input with a new part_type. Using kwargs indirection to
    # avoid a TypeError at import time if the field is not yet present — the
    # more targeted ``test_edit_damage_input_exposes_part_type_field`` will
    # have already flagged that, and this behavioural test adds defence in
    # depth once the field is introduced.
    try:
        input_data = EditDamageInput(  # type: ignore[call-arg]
            damage_id=damage.id,
            damage_type=DamageType.SCRATCH,
            part_type=PartType.BUMPER,
        )
    except TypeError as exc:
        pytest.fail(
            "EditDamageInput does not accept a 'part_type' kwarg — spec §3 requires it. "
            f"Underlying error: {exc}"
        )

    await use_case.execute(input_data)

    stored = await repo.get(damage.id)
    assert stored is not None
    assert stored.part_type is PartType.BUMPER, (
        "Edit use case ran without error but did not update the damage's part_type. "
        "The 'Изменить принадлежность повреждения' requirement is therefore unmet."
    )
