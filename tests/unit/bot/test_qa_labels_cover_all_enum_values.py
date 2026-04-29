"""QA: user-facing labels must cover every enum value.

Both ``PART_LABELS`` and ``DAMAGE_LABELS`` are dict-lookup tables with a
``.get(value, value)`` fallback: when a label is missing, the bot silently
renders the raw snake_case enum value (e.g. ``"rear_windshield"``) to the
user. That's a regression trap — someone adds a new enum variant to match
the thesis spec (e.g. "crack → замена" workflow), wires it into the
pricing rules and the keyboards, but forgets the labels dict, and the
end user starts seeing English tokens instead of Russian labels.

These tests are contract tests — they fail the moment enum and labels
drift out of sync, which is exactly the behaviour the bot UX depends on.
"""

from __future__ import annotations

import pytest

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType
from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


def test_every_part_type_has_a_russian_label() -> None:
    """PART_LABELS must contain an entry for every ``PartType`` member, or
    the inline part-selection keyboard will render raw tokens on at least
    one button."""

    missing = [p.value for p in PartType if p.value not in PART_LABELS]
    assert not missing, (
        f"PartType members without a label: {missing}. Extend PART_LABELS "
        f"in bot/labels.py — otherwise these buttons will show raw snake_case "
        f"tokens in VK."
    )


def test_every_damage_type_has_a_russian_label() -> None:
    missing = [d.value for d in DamageType if d.value not in DAMAGE_LABELS]
    assert not missing, (
        f"DamageType members without a label: {missing}. Extend DAMAGE_LABELS "
        f"in bot/labels.py — otherwise the pricing breakdown and the edit "
        f"keyboard will leak internal identifiers."
    )


def test_label_dicts_do_not_contain_orphan_entries() -> None:
    """Inverse direction: a label for a non-existent enum value is dead
    code that still gets shipped to users if a typo matches someone's
    payload. Drop it."""

    part_orphans = [k for k in PART_LABELS if k not in {p.value for p in PartType}]
    damage_orphans = [k for k in DAMAGE_LABELS if k not in {d.value for d in DamageType}]
    assert not part_orphans, f"PART_LABELS has orphan keys not in PartType: {part_orphans}"
    assert not damage_orphans, f"DAMAGE_LABELS has orphan keys not in DamageType: {damage_orphans}"


@pytest.mark.parametrize("part_value", [p.value for p in PartType])
def test_part_label_is_non_empty_russian_string(part_value: str) -> None:
    """Defence against ``PART_LABELS["door"] = ""`` accidents — the lookup
    would succeed but the user would see empty button text."""

    label = PART_LABELS[part_value]
    assert label.strip(), f"Empty label for PartType={part_value!r}"


@pytest.mark.parametrize("damage_value", [d.value for d in DamageType])
def test_damage_label_is_non_empty_russian_string(damage_value: str) -> None:
    label = DAMAGE_LABELS[damage_value]
    assert label.strip(), f"Empty label for DamageType={damage_value!r}"
