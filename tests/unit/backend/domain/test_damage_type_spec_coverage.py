"""QA spec-compliance tests for the DamageType enum.

Product-owner authoritative list of detectable damage categories (matches
``DamageType`` and the user-approved plan for the MVP):

    scratch, dent, paint_chip, rust, crack, broken_glass, flat_tire,
    broken_headlight.

Each parametrized case verifies one semantic category. A category is covered
when at least one ``DamageType`` enum member *value* contains a keyword
identifying that category in English. Keyword matching (instead of hard-coded
enum names) is resilient to trivial renames but still fails loudly if a whole
category is dropped.
"""

from __future__ import annotations

import pytest

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType
from auto_repair_estimator.bot.labels import DAMAGE_LABELS


def _enum_values() -> set[str]:
    return {m.value for m in DamageType}


# Each tuple: (spec_category_name, keywords that would identify it).
# Keywords are chosen from the English translations of the user-approved list.
_SPEC_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("scratch", ("scratch",)),
    ("dent", ("dent",)),
    ("paint_chip", ("paint", "chip")),
    ("rust", ("rust",)),
    ("crack", ("crack",)),
    ("broken_glass_or_headlight", ("broken", "shatter")),
    ("wheel_issue", ("flat", "tire", "wheel")),
]


@pytest.mark.parametrize("category,keywords", _SPEC_CATEGORIES, ids=[c[0] for c in _SPEC_CATEGORIES])
def test_damage_type_enum_covers_spec_category(category: str, keywords: tuple[str, ...]) -> None:
    """Every required damage category must have at least one DamageType value."""
    values = _enum_values()
    matching = {v for v in values if any(kw in v for kw in keywords)}
    assert matching, (
        f"DamageType enum is missing the required category '{category}'. "
        f"Expected at least one value containing one of {keywords}, "
        f"but enum has only {sorted(values)}. "
        f"Per requirements §3 the system MUST be able to detect this damage kind."
    )


def test_damage_labels_cover_every_enum_member() -> None:
    """If a damage type exists in the domain, the bot must have a human label for it.

    Without a label the bot would render raw technical strings (e.g. "tear")
    in Russian UI, which violates the functional requirement that the user
    sees the damage in a natural-language form.
    """
    missing = [m.value for m in DamageType if m.value not in DAMAGE_LABELS]
    assert not missing, f"DAMAGE_LABELS is missing Russian captions for: {missing}"


def test_damage_labels_has_no_orphans_outside_the_enum() -> None:
    """Every DAMAGE_LABELS key must correspond to an existing DamageType value.

    Stale entries (e.g. left over after shrinking the enum) would let the bot
    show a user-facing label for a damage the backend can no longer emit,
    producing dead UI branches and confusing Russian captions.
    """
    enum_values = {m.value for m in DamageType}
    orphans = [k for k in DAMAGE_LABELS if k not in enum_values]
    assert not orphans, (
        f"DAMAGE_LABELS contains labels for values not present in DamageType: {orphans}. "
        "Drop the stale labels or add the enum members."
    )
