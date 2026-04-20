"""QA: PricingService must inform the user when a damage has no rule.

The rate card (``pricing_rules`` seed in ``docker/init.sql``) deliberately
omits physically impossible combinations such as ``door + broken_glass``,
``roof + flat_tire``, ``side_window + scratch``, etc. In the bot UI,
however, the user can pick ANY combination through the inline keyboards:
`part_selection` lists all 12 parts and `damage_type_selection` lists all
8 damage types.

When an unpriceable combination reaches ``PricingService.calculate`` today
it is silently dropped (``logger.warning("No pricing rule … — skipping")``
is internal-only). The user then sees either

* a confusing total (other damages priced, this one missing, no explanation),
* or an empty breakdown with just "Кузовной ремонт не требуется", even when
  they explicitly asked for a scratch on glass.

Business contract: if the service skips a damage, it must surface a
user-visible ``note`` so the bot can explain what happened.

The tests below isolate three exemplary unpriceable combinations and assert
the note emerges. They also assert that the note mentions both the part
and the damage so the user can act on it.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.pricing_rule import PricingRule
from auto_repair_estimator.backend.domain.services.pricing_service import PricingService
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
)


class _SparsePricingRuleRepo:
    """Repo that only contains the rules we pass in explicitly."""

    def __init__(self, rules: list[PricingRule]) -> None:
        self._rules = {(r.part_type, r.damage_type): r for r in rules}

    async def get_rule(self, part_type: PartType, damage_type: DamageType) -> PricingRule | None:
        return self._rules.get((part_type, damage_type))

    async def get_all(self) -> list[PricingRule]:
        return list(self._rules.values())


def _make_damage(part: PartType, damage: DamageType) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-1",
        damage_type=damage,
        part_type=part,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )


# Three realistic unpriceable pairs coming out of the bot's inline keyboards.
# The labels are only used in pytest IDs for readability.
_UNPRICEABLE_PAIRS: list[tuple[PartType, DamageType, str, str]] = [
    (PartType.DOOR, DamageType.BROKEN_GLASS, "door", "broken_glass"),
    (PartType.ROOF, DamageType.FLAT_TIRE, "roof", "flat_tire"),
    (PartType.SIDE_WINDOW, DamageType.SCRATCH, "side_window", "scratch"),
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "part,damage,part_token,damage_token",
    _UNPRICEABLE_PAIRS,
    ids=[f"{p}-{d}" for _, _, p, d in _UNPRICEABLE_PAIRS],
)
async def test_unpriceable_combination_emits_a_user_visible_note(
    part: PartType,
    damage: DamageType,
    part_token: str,
    damage_token: str,
) -> None:
    """When a damage has no matching pricing rule, the service must add
    a ``note`` that mentions the part and damage so the bot can surface
    a coherent explanation to the user instead of silently dropping the
    line item."""

    # Arrange — repo intentionally empty so the lookup always returns None.
    service = PricingService(_rule_repository=_SparsePricingRuleRepo(rules=[]))

    # Act
    result = await service.calculate("req-1", [_make_damage(part, damage)])

    # Assert — at least one note references BOTH the part and the damage,
    # or a clear generic "no rule" explanation.
    notes_joined = " ".join(result.notes).lower()
    mentions_part = part_token in notes_joined or part.value in notes_joined
    mentions_damage = damage_token in notes_joined or damage.value in notes_joined
    generic_no_rule = "не оценива" in notes_joined or "нет тариф" in notes_joined or "no pricing" in notes_joined

    assert result.notes, (
        f"PricingService silently dropped {part.value}+{damage.value} with no user-visible note. "
        "The user will see an empty breakdown and no explanation for why their chosen damage "
        "was ignored. Add a note when get_rule returns None."
    )
    assert (mentions_part and mentions_damage) or generic_no_rule, (
        f"PricingService produced notes {result.notes!r} but none of them identifies "
        f"the unpriceable pair ({part.value}, {damage.value}). The user cannot act on the "
        "message."
    )
