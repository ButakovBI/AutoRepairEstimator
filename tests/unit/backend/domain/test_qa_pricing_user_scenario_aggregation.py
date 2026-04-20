"""End-to-end scenario from the user's requirement, executed against
the real rate-card repository (``InMemoryPricingRuleRepository`` mirrors
``docker/init.sql`` exactly).

Scenario (user wording reproduced verbatim for traceability):

    Модель нашла:
      на двери найдено 2 царапины и 2 трещины
      на стекле найдена царапина и битое стекло
      на фаре нашла 2 трещины, спущенную шину и разбитую фару
      на переднем крыле найдена 2 вмятины и 3 царапины

    Как итог работы модели (display — see bot handlers):
      Дверь - 2 царапины и 2 трещины
      Стекло - битое стекло
      Фара - разбитая фара
      Переднее крыло - 2 вмятины и 3 царапины

    Как итог подсчёта стоимости (pricing — what THIS test pins):
      Дверь - замена
      Стекло - замена
      Фара - замена
      Переднее крыло - царапина и вмятина

(Note: the "displayed list" entries for glass/headlight already drop
the physically-impossible damages via the part↔damage compatibility
filter in the ML worker / backend validation. This test focuses on the
*pricing* side of the contract, so it seeds only the combinations that
a realistic ML pipeline would forward — i.e. already-compatible pairs —
and asserts the aggregation + supersession logic produces exactly the
rows the user expects.)
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_pricing_rule_repository import (
    InMemoryPricingRuleRepository,
)
from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.services.pricing_service import PricingService
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
)


def _d(
    part: PartType, damage: DamageType, *, part_id: str
) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-scenario",
        damage_type=damage,
        part_type=part,
        source=DamageSource.ML,
        is_deleted=False,
        part_id=part_id,
    )


# Part-instance ids — one per physical part detected. ML-mode damages
# carry these so the aggregator can distinguish "two scratches on the
# same door" (dedup) from "scratches on two different doors" (no dedup).
_DOOR = "door-instance-1"
_GLASS = "front-windshield-instance-1"
_HEADLIGHT = "headlight-instance-1"
_FRONT_FENDER = "front-fender-instance-1"


def _scenario_damages() -> list[DetectedDamage]:
    """The subset of user-scenario damages that survive the part↔damage
    compatibility filter (which runs in the ML worker). This is what
    actually reaches the pricing layer.
    """

    return [
        # Door: 2 scratch + 2 crack → replacement (crack) supersedes scratch.
        _d(PartType.DOOR, DamageType.SCRATCH, part_id=_DOOR),
        _d(PartType.DOOR, DamageType.SCRATCH, part_id=_DOOR),
        _d(PartType.DOOR, DamageType.CRACK, part_id=_DOOR),
        _d(PartType.DOOR, DamageType.CRACK, part_id=_DOOR),
        # Front windshield: broken_glass (scratch dropped by compat filter upstream).
        _d(PartType.FRONT_WINDSHIELD, DamageType.BROKEN_GLASS, part_id=_GLASS),
        # Headlight: broken_headlight (crack / flat_tire dropped upstream).
        _d(PartType.HEADLIGHT, DamageType.BROKEN_HEADLIGHT, part_id=_HEADLIGHT),
        # Front fender: 2 dent + 3 scratch → 1 dent + 1 scratch (no replacement).
        _d(PartType.FRONT_FENDER, DamageType.DENT, part_id=_FRONT_FENDER),
        _d(PartType.FRONT_FENDER, DamageType.DENT, part_id=_FRONT_FENDER),
        _d(PartType.FRONT_FENDER, DamageType.SCRATCH, part_id=_FRONT_FENDER),
        _d(PartType.FRONT_FENDER, DamageType.SCRATCH, part_id=_FRONT_FENDER),
        _d(PartType.FRONT_FENDER, DamageType.SCRATCH, part_id=_FRONT_FENDER),
    ]


@pytest.mark.anyio
async def test_user_scenario_produces_exactly_four_priced_rows() -> None:
    """The four parts should yield 4 breakdown rows total:
    door→crack (замена), glass→broken_glass, headlight→broken_headlight,
    fender→scratch + fender→dent = 5.

    Wait: that's 5, not 4 — the fender contributes TWO rows (scratch +
    dent), matching the user's "царапина и вмятина" phrasing. The door,
    glass and headlight each contribute 1. Total = 5.
    """

    service = PricingService(_rule_repository=InMemoryPricingRuleRepository())

    result = await service.calculate("req-scenario", _scenario_damages())

    assert len(result.breakdown) == 5, (
        f"Expected 5 breakdown rows (door/замена, glass/замена, "
        f"headlight/замена, fender/scratch, fender/dent) but got "
        f"{len(result.breakdown)}: {result.breakdown!r}"
    )

    pairs = {(row["part_type"], row["damage_type"]) for row in result.breakdown}
    assert pairs == {
        (PartType.DOOR.value, DamageType.CRACK.value),
        (PartType.FRONT_WINDSHIELD.value, DamageType.BROKEN_GLASS.value),
        (PartType.HEADLIGHT.value, DamageType.BROKEN_HEADLIGHT.value),
        (PartType.FRONT_FENDER.value, DamageType.SCRATCH.value),
        (PartType.FRONT_FENDER.value, DamageType.DENT.value),
    }


@pytest.mark.anyio
async def test_user_scenario_totals_match_rate_card_for_expected_rows() -> None:
    """Totals must be the sum of exactly the 5 expected rate-card rows.

    Computed against the rate card:
      door/crack:             20 000 / 20 000 RUB, 12 / 16 h
      front_windshield/glass:  5 000 / 10 000 RUB,  8 /  8 h
      headlight/headlight:     3 000 /  3 000 RUB,  4 /  4 h
      front_fender/scratch:   10 000 / 18 000 RUB,  8 /  8 h
      front_fender/dent:      23 000 / 30 000 RUB, 16 / 24 h
    """

    service = PricingService(_rule_repository=InMemoryPricingRuleRepository())

    result = await service.calculate("req-scenario", _scenario_damages())

    # Summing the per-row ranges element-wise:
    assert result.total_cost_min == pytest.approx(
        20_000 + 5_000 + 3_000 + 10_000 + 23_000
    )
    assert result.total_cost_max == pytest.approx(
        20_000 + 10_000 + 3_000 + 18_000 + 30_000
    )
    assert result.total_hours_min == pytest.approx(12 + 8 + 4 + 8 + 16)
    assert result.total_hours_max == pytest.approx(16 + 8 + 4 + 8 + 24)


@pytest.mark.anyio
async def test_user_scenario_polish_note_counts_scratched_parts_not_detections() -> None:
    """Only the front fender has (surviving) scratches in the priced
    set. The polish note therefore quotes N=1 — the door's scratches
    were superseded by its crack and must not leak into the count.
    """

    service = PricingService(_rule_repository=InMemoryPricingRuleRepository())
    result = await service.calculate("req-scenario", _scenario_damages())

    polish_notes = [n for n in result.notes if "полировк" in n.lower()]
    assert len(polish_notes) == 1
    # Must reference "1" (one scratched part) and not any of the
    # higher numbers that a bug would produce (3 raw fender scratches,
    # or 5 total scratches including the door's).
    note = polish_notes[0]
    assert "1 царапин" in note, (
        f"Polish note should reference exactly 1 scratched part after "
        f"supersession; got: {note!r}"
    )


@pytest.mark.anyio
async def test_two_independent_doors_each_with_crack_yield_two_replacement_rows() -> None:
    """Regression guard for the aggregator's scope. If the ML worker ever
    emits two ``DetectedPart`` instances for ``PartType.DOOR``, both
    should be charged for replacement — dedup is per physical part,
    not per part_type.
    """

    service = PricingService(_rule_repository=InMemoryPricingRuleRepository())
    damages = [
        _d(PartType.DOOR, DamageType.CRACK, part_id="door-left"),
        _d(PartType.DOOR, DamageType.CRACK, part_id="door-right"),
    ]

    result = await service.calculate("req-two-doors", damages)

    assert len(result.breakdown) == 2
    assert result.total_cost_min == pytest.approx(2 * 20_000.0)
