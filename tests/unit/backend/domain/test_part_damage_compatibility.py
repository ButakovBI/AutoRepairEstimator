"""Contract tests for PART_DAMAGE_COMPATIBILITY.

This file protects the SSOT invariant: every `PartType` enum value is
registered with a non-empty set of valid `DamageType` values, and the
mapping stays aligned with the pricing rules seeded in
``InMemoryPricingRuleRepository`` (which mirrors ``docker/init.sql``).

Without these tests a future refactor could add a new `PartType`, forget
to register it here, and the bot's damage-type keyboard would silently
degrade to an empty keyboard for that part.
"""

from __future__ import annotations

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_pricing_rule_repository import (
    InMemoryPricingRuleRepository,
)
from auto_repair_estimator.backend.domain.value_objects.part_damage_compatibility import (
    PART_DAMAGE_COMPATIBILITY,
    compatible_damages_for,
    is_compatible_pair,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType


class TestCompatibilityShape:
    def test_every_part_type_is_registered(self):
        missing = set(PartType) - set(PART_DAMAGE_COMPATIBILITY)
        assert not missing, f"PartType members missing from SSOT: {sorted(p.value for p in missing)}"

    def test_no_part_has_an_empty_damage_set(self):
        # An empty set would mean "this part has no business in the shop" --
        # every registered part must have at least one priceable damage type.
        empty = [p.value for p, dmgs in PART_DAMAGE_COMPATIBILITY.items() if not dmgs]
        assert empty == []

    def test_every_damage_value_is_a_valid_enum_member(self):
        all_damage = {d for dmgs in PART_DAMAGE_COMPATIBILITY.values() for d in dmgs}
        assert all_damage <= set(DamageType)


class TestCompatibilityBusinessRules:
    @pytest.mark.parametrize(
        "part,expected",
        [
            (PartType.DOOR, {"scratch", "dent", "paint_chip", "rust", "crack"}),
            (PartType.BUMPER, {"scratch", "dent", "paint_chip", "rust", "crack"}),
            (PartType.HOOD, {"scratch", "dent", "paint_chip", "rust", "crack"}),
            (PartType.HEADLIGHT, {"broken_headlight"}),
            (PartType.WHEEL, {"flat_tire"}),
            (PartType.FRONT_WINDSHIELD, {"broken_glass"}),
            (PartType.REAR_WINDSHIELD, {"broken_glass"}),
            (PartType.SIDE_WINDOW, {"broken_glass"}),
        ],
    )
    def test_known_part_has_expected_damage_set(self, part: PartType, expected: set[str]):
        actual = {d.value for d in compatible_damages_for(part)}
        assert actual == expected

    def test_is_compatible_pair_accepts_known_good(self):
        assert is_compatible_pair(PartType.HOOD, DamageType.SCRATCH) is True

    def test_is_compatible_pair_rejects_headlight_scratch(self):
        # The specific bug the user reported: scratch on a headlight used to
        # show as a valid button even though the shop only prices replacement.
        assert is_compatible_pair(PartType.HEADLIGHT, DamageType.SCRATCH) is False

    def test_is_compatible_pair_rejects_wheel_dent(self):
        assert is_compatible_pair(PartType.WHEEL, DamageType.DENT) is False

    def test_is_compatible_pair_rejects_door_broken_glass(self):
        assert is_compatible_pair(PartType.DOOR, DamageType.BROKEN_GLASS) is False


class TestCompatibilityMatchesPricingRules:
    """The SSOT must stay aligned with the priced-pair table.

    The only documented exception is wheel/flat_tire -- priced via
    ``TYRE_SHOP_NOTE`` rather than a ``pricing_rules`` row. Every OTHER
    compatible pair MUST have a matching row; every priced row MUST
    correspond to a compatible pair.
    """

    async def test_every_compatible_pair_except_wheel_has_pricing_rule(self):
        repo = InMemoryPricingRuleRepository()
        missing: list[tuple[str, str]] = []
        for part, dmgs in PART_DAMAGE_COMPATIBILITY.items():
            if part is PartType.WHEEL:
                # wheel -> tyre shop; no pricing_rules row by design
                continue
            for dmg in dmgs:
                rule = await repo.get_rule(part, dmg)
                if rule is None:
                    missing.append((part.value, dmg.value))
        assert missing == [], f"compatible pairs without pricing rule: {missing}"

    async def test_every_pricing_rule_corresponds_to_compatible_pair(self):
        repo = InMemoryPricingRuleRepository()
        all_rules = await repo.get_all()
        orphaned: list[tuple[str, str]] = [
            (r.part_type.value, r.damage_type.value)
            for r in all_rules
            if not is_compatible_pair(r.part_type, r.damage_type)
        ]
        assert orphaned == [], f"priced pairs not in SSOT: {orphaned}"
