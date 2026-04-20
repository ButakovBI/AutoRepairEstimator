"""Direct unit tests for :func:`aggregate_damages_for_pricing`.

Keeps the aggregator's business rules independently testable from the
:class:`PricingService` so a regression in either layer is pinned to
the right place.
"""

from __future__ import annotations

from uuid import uuid4

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.services.damage_aggregator import (
    aggregate_damages_for_pricing,
)
from auto_repair_estimator.backend.domain.value_objects.damage_severity import (
    REPLACEMENT_DAMAGE_TYPES,
    causes_replacement,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
)


def _d(
    part: PartType,
    damage: DamageType,
    *,
    part_id: str | None = None,
    source: DamageSource = DamageSource.ML,
) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-agg",
        damage_type=damage,
        part_type=part,
        source=source,
        is_deleted=False,
        part_id=part_id,
    )


class TestPerPartDamageTypeDedup:
    def test_three_scratches_on_one_door_collapse_to_one(self) -> None:
        """The concrete example from the user spec: don't double-charge
        painting for multiple scratches on the same physical door."""

        door_id = "door-1"
        damages = [
            _d(PartType.DOOR, DamageType.SCRATCH, part_id=door_id),
            _d(PartType.DOOR, DamageType.SCRATCH, part_id=door_id),
            _d(PartType.DOOR, DamageType.SCRATCH, part_id=door_id),
        ]
        result = aggregate_damages_for_pricing(damages)

        assert len(result.kept) == 1
        assert result.kept[0].damage_type is DamageType.SCRATCH
        assert result.dropped_duplicates == 2

    def test_two_dents_plus_three_scratches_on_fender_yields_dent_plus_scratch(
        self,
    ) -> None:
        """From the user scenario: "переднее крыло - 2 вмятины и 3 царапины"
        must price as one dent + one scratch (both non-replacement, so
        both survive dedup).
        """

        fender_id = "fender-front"
        damages = [
            _d(PartType.FRONT_FENDER, DamageType.DENT, part_id=fender_id),
            _d(PartType.FRONT_FENDER, DamageType.DENT, part_id=fender_id),
            _d(PartType.FRONT_FENDER, DamageType.SCRATCH, part_id=fender_id),
            _d(PartType.FRONT_FENDER, DamageType.SCRATCH, part_id=fender_id),
            _d(PartType.FRONT_FENDER, DamageType.SCRATCH, part_id=fender_id),
        ]
        result = aggregate_damages_for_pricing(damages)

        kept_types = {d.damage_type for d in result.kept}
        assert kept_types == {DamageType.DENT, DamageType.SCRATCH}
        # 5 input, 2 unique types → 3 dropped as duplicates.
        assert result.dropped_duplicates == 3
        assert result.dropped_by_replacement_supersession == 0

    def test_manual_damages_without_part_id_dedup_by_part_type(self) -> None:
        """Manual-mode damages never carry a ``part_id`` because the
        user refers to "the door" abstractly. The aggregator must still
        dedup them — otherwise a user adding the same pair twice would
        be double-charged.
        """

        damages = [
            _d(PartType.HOOD, DamageType.DENT, source=DamageSource.MANUAL),
            _d(PartType.HOOD, DamageType.DENT, source=DamageSource.MANUAL),
        ]
        result = aggregate_damages_for_pricing(damages)

        assert len(result.kept) == 1
        assert result.dropped_duplicates == 1

    def test_different_part_instances_same_type_do_not_dedup(self) -> None:
        """Two different doors with scratch → two rows. If the ML worker
        ever distinguishes door instances via ``part_id``, they must be
        billed separately.
        """

        damages = [
            _d(PartType.DOOR, DamageType.SCRATCH, part_id="door-left"),
            _d(PartType.DOOR, DamageType.SCRATCH, part_id="door-right"),
        ]
        result = aggregate_damages_for_pricing(damages)

        assert len(result.kept) == 2
        assert result.dropped_duplicates == 0


class TestReplacementSupersession:
    def test_door_with_scratch_and_crack_collapses_to_one_replacement_row(
        self,
    ) -> None:
        """From user scenario: "дверь — 2 царапины и 2 трещины → замена".
        The crack triggers replacement, painting drops out.
        """

        door_id = "door-1"
        damages = [
            _d(PartType.DOOR, DamageType.SCRATCH, part_id=door_id),
            _d(PartType.DOOR, DamageType.SCRATCH, part_id=door_id),
            _d(PartType.DOOR, DamageType.CRACK, part_id=door_id),
            _d(PartType.DOOR, DamageType.CRACK, part_id=door_id),
        ]
        result = aggregate_damages_for_pricing(damages)

        assert len(result.kept) == 1
        assert result.kept[0].damage_type is DamageType.CRACK
        # 4 → 2 distinct types after dedup (scratch, crack), then +1 drop
        # for scratch being superseded by crack.
        assert result.dropped_duplicates == 2
        assert result.dropped_by_replacement_supersession == 1

    def test_paint_chip_and_crack_on_same_part_collapse_to_one_row(self) -> None:
        """Two different replacement-class damages on the same part also
        collapse to a single row — you replace the part once, not twice.
        """

        damages = [
            _d(PartType.DOOR, DamageType.PAINT_CHIP, part_id="d1"),
            _d(PartType.DOOR, DamageType.CRACK, part_id="d1"),
        ]
        result = aggregate_damages_for_pricing(damages)

        assert len(result.kept) == 1
        assert causes_replacement(result.kept[0].damage_type)

    def test_non_replacement_damages_are_preserved_when_no_replacement_present(
        self,
    ) -> None:
        """Sanity check: without a replacement damage, both types survive."""

        damages = [
            _d(PartType.DOOR, DamageType.SCRATCH, part_id="d1"),
            _d(PartType.DOOR, DamageType.DENT, part_id="d1"),
        ]
        result = aggregate_damages_for_pricing(damages)

        kept_types = {d.damage_type for d in result.kept}
        assert kept_types == {DamageType.SCRATCH, DamageType.DENT}
        assert result.dropped_by_replacement_supersession == 0

    def test_replacement_on_one_part_does_not_affect_other_parts(self) -> None:
        """Supersession is scoped per part. A crack on the door must not
        silence a scratch on the hood.
        """

        damages = [
            _d(PartType.DOOR, DamageType.CRACK, part_id="door-1"),
            _d(PartType.DOOR, DamageType.SCRATCH, part_id="door-1"),
            _d(PartType.HOOD, DamageType.SCRATCH, part_id="hood-1"),
        ]
        result = aggregate_damages_for_pricing(damages)

        # Door collapses to one crack; hood keeps its scratch.
        kept_pairs = {(d.part_type, d.damage_type) for d in result.kept}
        assert kept_pairs == {
            (PartType.DOOR, DamageType.CRACK),
            (PartType.HOOD, DamageType.SCRATCH),
        }


class TestReplacementDamageTypesContract:
    def test_replacement_set_matches_documented_constants(self) -> None:
        """Contract guard: if someone adds a new replacement-class damage
        to the enum, the SSOT set must be updated too — otherwise pricing
        silently misses the new case. We assert the EXACT set so any
        drift is caught immediately.
        """

        assert REPLACEMENT_DAMAGE_TYPES == frozenset(
            {
                DamageType.PAINT_CHIP,
                DamageType.CRACK,
                DamageType.BROKEN_GLASS,
                DamageType.BROKEN_HEADLIGHT,
            }
        )

    def test_replacement_and_non_replacement_are_disjoint(self) -> None:
        """Every damage type is either replacement or not — no overlap."""

        non_replacement = {
            DamageType.SCRATCH,
            DamageType.DENT,
            DamageType.RUST,
            DamageType.FLAT_TIRE,
        }
        assert non_replacement.isdisjoint(REPLACEMENT_DAMAGE_TYPES)

    def test_all_damage_types_are_classified(self) -> None:
        """Every DamageType must belong to exactly one of the two groups.
        If a new enum value is added without a home, this test fails loudly.
        """

        non_replacement = {
            DamageType.SCRATCH,
            DamageType.DENT,
            DamageType.RUST,
            DamageType.FLAT_TIRE,
        }
        all_known = non_replacement | REPLACEMENT_DAMAGE_TYPES
        missing = set(DamageType) - all_known
        assert not missing, (
            f"DamageType(s) {missing} are not classified as replacement "
            "or non-replacement. Update REPLACEMENT_DAMAGE_TYPES or the "
            "non-replacement set in this test."
        )
