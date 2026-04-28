"""Aggregate :class:`DetectedDamage` instances into priced representatives.

Business rules (per thesis requirements, confirmed by user):

1. **Physical-part dedup by damage type.** If one part instance has
   several damages of the same ``damage_type`` (e.g. three scratches on
   the same door detected as separate ML boxes, or a user adds
   ``door + dent`` twice manually), price it as a *single* occurrence —
   you paint a door once regardless of how many scratches it carries.

2. **Replacement supersedes other repairs on the same part.** If a part
   has both a replacement-class damage (paint_chip / crack /
   broken_glass / broken_headlight) AND non-replacement damages
   (scratch / dent / rust), the replacement subsumes the others: you
   don't paint a panel that's being replaced. We collapse the part to
   exactly one "замена" row.

3. **Wheel damages are not touched here** — the pricing service routes
   them to a tyre-shop note out-of-band before aggregation runs.

"Physical part" is identified by :attr:`DetectedDamage.part_id` when
present (ML mode persists one :class:`DetectedPart` per detected
instance). For manual-mode damages ``part_id`` is ``None``; there is no
way to distinguish "the user meant two different door panels" from
"the user added the same pair twice", so we fall back to grouping by
``part_type``. This biases towards de-duplication, which matches the
bot-UI contract (each ``part_type`` appears as a single selection in
the keyboard).

The aggregator is a pure function over domain entities — no repository
or I/O coupling — so it is trivially unit-testable and reused by
:class:`PricingService` without an injection seam.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.value_objects.damage_severity import (
    causes_replacement,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageType,
    PartType,
)

# Grouping key: the part instance id if we have one, else the part type.
# Using a tagged tuple keeps the two namespaces disjoint so a manual
# ``door`` (key=("type", PartType.DOOR)) never collides with an ML
# ``door`` instance id (key=("id", "uuid-...")).
_PartKey = tuple[str, str | PartType]


def _part_key(damage: DetectedDamage) -> _PartKey:
    if damage.part_id is not None:
        return ("id", damage.part_id)
    return ("type", damage.part_type)


@dataclass(frozen=True)
class AggregationBreakdown:
    """Diagnostic info about what the aggregator dropped.

    Exposed so ``PricingService`` (or a future observability sink) can
    report counts in logs without re-computing them. Not part of any
    public HTTP contract today.
    """

    kept: list[DetectedDamage]
    dropped_duplicates: int
    dropped_by_replacement_supersession: int


def aggregate_damages_for_pricing(
    damages: Iterable[DetectedDamage],
) -> AggregationBreakdown:
    """Return the damages that should be priced, with diagnostics.

    Deleted damages are passed through *unfiltered* here — the pricing
    service filters them out earlier. Keeping this function responsible
    only for the aggregation semantics keeps it easy to reason about.
    """

    # First pass: for each part, keep an ordered-by-first-seen map of
    # damage_type -> representative damage. Using OrderedDict preserves
    # the original detection order, so the resulting breakdown rows
    # line up with the order the ML worker produced them (useful for
    # visual parity between the "found damages" list and the priced
    # breakdown).
    per_part: dict[_PartKey, OrderedDict[DamageType, DetectedDamage]] = {}
    dup_count = 0

    for damage in damages:
        key = _part_key(damage)
        bucket = per_part.setdefault(key, OrderedDict())
        if damage.damage_type in bucket:
            dup_count += 1
            continue
        bucket[damage.damage_type] = damage

    # Second pass: replacement supersession. For each part, if any
    # representative causes replacement, keep exactly one replacement
    # representative and drop the non-replacement ones.
    kept: list[DetectedDamage] = []
    replacement_drops = 0

    for bucket in per_part.values():
        replacement_reps = [
            d for d in bucket.values() if causes_replacement(d.damage_type)
        ]
        if replacement_reps:
            # Collapse all replacement reps into one — they all price
            # identically for any given part in the rate card (see
            # ``REPLACEMENT_DAMAGE_TYPES`` docstring). We keep the first
            # one in detection order for stability, and count the rest
            # as "superseded" (includes non-replacement reps on the
            # same part plus extra replacement reps).
            chosen = replacement_reps[0]
            dropped_here = len(bucket) - 1
            replacement_drops += dropped_here
            kept.append(chosen)
        else:
            kept.extend(bucket.values())

    return AggregationBreakdown(
        kept=kept,
        dropped_duplicates=dup_count,
        dropped_by_replacement_supersession=replacement_drops,
    )


__all__ = ["AggregationBreakdown", "aggregate_damages_for_pricing"]


# Tiny belt-and-braces: make PartType / DamageType importers don't remove
# the imports if the implementation above is ever inlined elsewhere.
_ = (PartType, DamageType)
