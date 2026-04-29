"""Which :class:`DamageType` values imply full-part replacement.

Single source of truth, derived from the shop rate card (thesis tables
5 & 6, also mirrored in ``docker/init.sql`` and
:class:`InMemoryPricingRuleRepository`). The pricing rules in the rate
card map these four damage classes to the "замена детали" treatment
column, priced identically to each other within a given part:

* :data:`DamageType.PAINT_CHIP`        — "отвалившийся кусок краски"
* :data:`DamageType.CRACK`             — "трещина"
* :data:`DamageType.BROKEN_GLASS`      — "разбитое стекло"
* :data:`DamageType.BROKEN_HEADLIGHT`  — "разбитая фара"

Why extract this as a value object (instead of inspecting pricing rule
labels at runtime):

* Pricing rules live in the rule repository; they don't carry a
  "treatment" column, only numeric ranges. Inferring "this is a
  replacement" from numeric equality is brittle (some parts genuinely
  have the same body-work and replacement price, e.g. hood scratch and
  hood dent happen to align in some edge cases).
* The business rule that *replacement supersedes other repairs on the
  same part* is a domain fact, not a pricing implementation detail —
  it belongs at the value-object layer.
* Having a single frozenset keeps the pricing aggregator's hot loop
  allocation-free and trivially unit-testable.

A property-style contract test in ``test_qa_replacement_supersedes_other_damages.py``
asserts that for every body panel the replacement rows in the rate
card are exactly the rows keyed by these damage types — so if someone
later adds a new replacement-class damage to :class:`DamageType`
without updating this set, the test fails fast.
"""

from __future__ import annotations

from typing import Final

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType

REPLACEMENT_DAMAGE_TYPES: Final[frozenset[DamageType]] = frozenset(
    {
        DamageType.PAINT_CHIP,
        DamageType.CRACK,
        DamageType.BROKEN_GLASS,
        DamageType.BROKEN_HEADLIGHT,
    }
)


def causes_replacement(damage_type: DamageType) -> bool:
    """True iff ``damage_type`` implies replacing the part entirely.

    Use this over a raw ``in`` check at call sites that need to justify
    the branch in logs or docstrings — the named predicate makes intent
    obvious at the call site even without comments.
    """

    return damage_type in REPLACEMENT_DAMAGE_TYPES
