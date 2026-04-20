"""Single source of truth for which :class:`DamageType` values make sense on
which :class:`PartType`.

Before this module existed, compatibility knowledge was implicit in three
disconnected places:

* ``docker/init.sql`` / :mod:`.in_memory_pricing_rule_repository` — by the
  presence or absence of rows in ``pricing_rules``.
* :class:`PricingService` — via a ``wheel -> TYRE_SHOP_NOTE`` special case
  and a "no rule found" fallback that quietly attached a note.
* The bot UI — which **did not know**: it rendered all eight damage types
  for every part, letting the user pick physically impossible combinations
  (``flat_tire`` on a door, ``scratch`` on a headlight, …).

Centralising the mapping here means:

1. Bot keyboards (`damage_type_selection`, `edit_damage_type`) render only
   the buttons that will actually produce a priced row. No more dead-end
   choices.
2. Backend use cases (`AddDamage`, `EditDamage`) reject incompatible pairs
   with :class:`ValueError` (surfaced as HTTP 400). A client that bypasses
   the UI — old keyboard, manual ``curl``, replay attack — still can't
   pollute the damage table with nonsense.
3. A contract test guarantees this SSOT stays aligned with
   ``pricing_rules`` (the one documented exception is ``wheel → flat_tire``,
   which is intentionally routed to the tyre shop via a user note and has
   no pricing row).

Grouping rationale (derived from the thesis rate card — tables 5 & 6):

* Body panels receive classic surface/structural damage: scratch, dent,
  paint chip, rust, crack.
* Glass panels only have one failure mode: broken glass (cracks on glass
  are priced as ``broken_glass`` per the shop rate card).
* Headlights are treated as single assemblies: only ``broken_headlight``.
* Wheels are single assemblies pricing-wise: only ``flat_tire`` (routed
  out of the body shop via a note).
"""

from __future__ import annotations

from typing import Final

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType

# Body panels share the same five damage types: this is the rate-card
# "Кузовные детали" column of thesis table 5 applied uniformly.
_BODY_PANEL_DAMAGES: Final[frozenset[DamageType]] = frozenset(
    {
        DamageType.SCRATCH,
        DamageType.DENT,
        DamageType.PAINT_CHIP,
        DamageType.RUST,
        DamageType.CRACK,
    }
)

PART_DAMAGE_COMPATIBILITY: Final[dict[PartType, frozenset[DamageType]]] = {
    # --- body panels -------------------------------------------------------
    PartType.DOOR: _BODY_PANEL_DAMAGES,
    PartType.FRONT_FENDER: _BODY_PANEL_DAMAGES,
    PartType.REAR_FENDER: _BODY_PANEL_DAMAGES,
    PartType.TRUNK: _BODY_PANEL_DAMAGES,
    PartType.HOOD: _BODY_PANEL_DAMAGES,
    PartType.ROOF: _BODY_PANEL_DAMAGES,
    PartType.BUMPER: _BODY_PANEL_DAMAGES,
    # --- glass: windshields + side window ---------------------------------
    PartType.FRONT_WINDSHIELD: frozenset({DamageType.BROKEN_GLASS}),
    PartType.REAR_WINDSHIELD: frozenset({DamageType.BROKEN_GLASS}),
    PartType.SIDE_WINDOW: frozenset({DamageType.BROKEN_GLASS}),
    # --- single-assembly parts --------------------------------------------
    PartType.HEADLIGHT: frozenset({DamageType.BROKEN_HEADLIGHT}),
    PartType.WHEEL: frozenset({DamageType.FLAT_TIRE}),
}


def compatible_damages_for(part_type: PartType) -> frozenset[DamageType]:
    """Return the :class:`DamageType` set allowed on ``part_type``.

    Raises :class:`KeyError` if ``part_type`` is a brand-new enum member
    that wasn't registered in :data:`PART_DAMAGE_COMPATIBILITY`. That's a
    *programmer* error, not a user error — the contract test in the unit
    suite enforces that every :class:`PartType` value is present here, so
    this path is only reachable during development before the test catches it.
    """

    return PART_DAMAGE_COMPATIBILITY[part_type]


def is_compatible_pair(part_type: PartType, damage_type: DamageType) -> bool:
    """True iff ``damage_type`` is a business-valid choice for ``part_type``."""

    return damage_type in PART_DAMAGE_COMPATIBILITY.get(part_type, frozenset())
