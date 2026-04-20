"""In-memory pricing rule repo, used in dev / test mode without Postgres.

The seed is a 1:1 mirror of ``docker/init.sql``, so running the bot
locally against the in-memory backend produces the same estimates the user
would get in production. Before this module existed, the dev fallback in
``backend/main.py`` returned ``None`` for every lookup, which silently
dropped every damage in ``PricingService`` and surfaced the misleading
"Кузовной ремонт по этой заявке не требуется" message for perfectly
priceable combinations (e.g. rust on trunk, paint chip on hood).

Damage-type → treatment mapping (as in the thesis spec):

* scratch      → покраска (table 5 "Царапина (покраска)" column)
* rust         → покраска (same column — same treatment cost/duration)
* dent         → рихтовка + покраска (table 5 "Вмятина")
* paint_chip   → замена детали (отвалившийся кусок краски → замена)
* crack        → замена детали (трещина → замена, для кузовных деталей)
* broken_glass       → замена детали (для стёкол)
* broken_headlight   → замена детали (для фары)
* flat_tire / любой на wheel → шиномонтаж (без priced-row, только текстовый
  note, выдаваемый через ``TYRE_SHOP_NOTE``)
"""

from __future__ import annotations

from typing import Final

from auto_repair_estimator.backend.domain.entities.pricing_rule import PricingRule
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType

# Hour conversions used to encode table 6 ("Длительность ремонта") into
# numeric labour hours. Matches the conversion spelled out in init.sql.
_DAY: Final[float] = 8.0
_HALF_DAY: Final[float] = 4.0
_HOUR: Final[float] = 1.0

_SEED: Final[list[PricingRule]] = [
    # Front fender: замена 20 / 1 день, вмятина 23-30 / 2-3 дня
    PricingRule(1, PartType.FRONT_FENDER, DamageType.SCRATCH, _DAY, _DAY, 10_000, 18_000),
    PricingRule(2, PartType.FRONT_FENDER, DamageType.RUST, _DAY, _DAY, 10_000, 18_000),
    PricingRule(3, PartType.FRONT_FENDER, DamageType.DENT, 2 * _DAY, 3 * _DAY, 23_000, 30_000),
    PricingRule(4, PartType.FRONT_FENDER, DamageType.PAINT_CHIP, _DAY, _DAY, 20_000, 20_000),
    PricingRule(5, PartType.FRONT_FENDER, DamageType.CRACK, _DAY, _DAY, 20_000, 20_000),
    # Rear fender: замена 75-100 / 5 дней
    PricingRule(6, PartType.REAR_FENDER, DamageType.SCRATCH, _DAY, _DAY, 10_000, 18_000),
    PricingRule(7, PartType.REAR_FENDER, DamageType.RUST, _DAY, _DAY, 10_000, 18_000),
    PricingRule(8, PartType.REAR_FENDER, DamageType.DENT, 2 * _DAY, 3 * _DAY, 23_000, 30_000),
    PricingRule(9, PartType.REAR_FENDER, DamageType.PAINT_CHIP, 5 * _DAY, 5 * _DAY, 75_000, 100_000),
    PricingRule(10, PartType.REAR_FENDER, DamageType.CRACK, 5 * _DAY, 5 * _DAY, 75_000, 100_000),
    # Door: замена 20 / 1.5-2 дня
    PricingRule(11, PartType.DOOR, DamageType.SCRATCH, _DAY, _DAY, 10_000, 18_000),
    PricingRule(12, PartType.DOOR, DamageType.RUST, _DAY, _DAY, 10_000, 18_000),
    PricingRule(13, PartType.DOOR, DamageType.DENT, 2 * _DAY, 3 * _DAY, 23_000, 30_000),
    PricingRule(14, PartType.DOOR, DamageType.PAINT_CHIP, 1.5 * _DAY, 2 * _DAY, 20_000, 20_000),
    PricingRule(15, PartType.DOOR, DamageType.CRACK, 1.5 * _DAY, 2 * _DAY, 20_000, 20_000),
    # Trunk: замена 20 / 1.5-2 дня
    PricingRule(16, PartType.TRUNK, DamageType.SCRATCH, _DAY, _DAY, 10_000, 18_000),
    PricingRule(17, PartType.TRUNK, DamageType.RUST, _DAY, _DAY, 10_000, 18_000),
    PricingRule(18, PartType.TRUNK, DamageType.DENT, 2 * _DAY, 3 * _DAY, 23_000, 30_000),
    PricingRule(19, PartType.TRUNK, DamageType.PAINT_CHIP, 1.5 * _DAY, 2 * _DAY, 20_000, 20_000),
    PricingRule(20, PartType.TRUNK, DamageType.CRACK, 1.5 * _DAY, 2 * _DAY, 20_000, 20_000),
    # Roof: замена 75-100 / 5 дней
    PricingRule(21, PartType.ROOF, DamageType.SCRATCH, _DAY, _DAY, 10_000, 18_000),
    PricingRule(22, PartType.ROOF, DamageType.RUST, _DAY, _DAY, 10_000, 18_000),
    PricingRule(23, PartType.ROOF, DamageType.DENT, 2 * _DAY, 3 * _DAY, 23_000, 30_000),
    PricingRule(24, PartType.ROOF, DamageType.PAINT_CHIP, 5 * _DAY, 5 * _DAY, 75_000, 100_000),
    PricingRule(25, PartType.ROOF, DamageType.CRACK, 5 * _DAY, 5 * _DAY, 75_000, 100_000),
    # Hood: вмятина 30-35 / 2 дня, замена 28-30 / 2 дня
    PricingRule(26, PartType.HOOD, DamageType.SCRATCH, _DAY, _DAY, 10_000, 18_000),
    PricingRule(27, PartType.HOOD, DamageType.RUST, _DAY, _DAY, 10_000, 18_000),
    PricingRule(28, PartType.HOOD, DamageType.DENT, 2 * _DAY, 2 * _DAY, 30_000, 35_000),
    PricingRule(29, PartType.HOOD, DamageType.PAINT_CHIP, 2 * _DAY, 2 * _DAY, 28_000, 30_000),
    PricingRule(30, PartType.HOOD, DamageType.CRACK, 2 * _DAY, 2 * _DAY, 28_000, 30_000),
    # Bumper: вмятина 3-5 / 1-2 дня, замена 18 / 1.5 дня
    PricingRule(31, PartType.BUMPER, DamageType.SCRATCH, _DAY, _DAY, 10_000, 18_000),
    PricingRule(32, PartType.BUMPER, DamageType.RUST, _DAY, _DAY, 10_000, 18_000),
    PricingRule(33, PartType.BUMPER, DamageType.DENT, _DAY, 2 * _DAY, 3_000, 5_000),
    PricingRule(34, PartType.BUMPER, DamageType.PAINT_CHIP, 1.5 * _DAY, 1.5 * _DAY, 18_000, 18_000),
    PricingRule(35, PartType.BUMPER, DamageType.CRACK, 1.5 * _DAY, 1.5 * _DAY, 18_000, 18_000),
    # Glass: только замена
    PricingRule(36, PartType.FRONT_WINDSHIELD, DamageType.BROKEN_GLASS, _DAY, _DAY, 5_000, 10_000),
    PricingRule(37, PartType.REAR_WINDSHIELD, DamageType.BROKEN_GLASS, _DAY, _DAY, 5_000, 10_000),
    PricingRule(38, PartType.SIDE_WINDOW, DamageType.BROKEN_GLASS, _DAY, _DAY, 3_000, 3_000),
    # Headlight: только замена
    PricingRule(39, PartType.HEADLIGHT, DamageType.BROKEN_HEADLIGHT, _HALF_DAY, _HALF_DAY, 3_000, 3_000),
    # Using _HOUR to document the polishing alternative stays non-zero, though
    # we don't store it here — ``pricing_constants.POLISH_*`` drives it.
    # Dummy binding to keep _HOUR referenced for future use.
    # (No rule is emitted for ``flat_tire``/wheel — those route to tyre shop.)
]

_SEED_SIZE_FOR_DOCS: Final[int] = len(_SEED)
_ = _HOUR  # Reserved for a future polishing-row if we ever lift the note into the table.


class InMemoryPricingRuleRepository:
    """A dev/test PricingRuleRepository pre-seeded with the thesis rate card.

    The class is intentionally stateless across instances — the seed is a
    module-level constant — but we still take a ``dict`` lookup hit once
    per instance so tests can subclass or monkey-patch ``_rules`` without
    leaking state into other tests.
    """

    def __init__(self) -> None:
        self._rules: dict[tuple[PartType, DamageType], PricingRule] = {
            (rule.part_type, rule.damage_type): rule for rule in _SEED
        }

    async def get_rule(self, part_type: PartType, damage_type: DamageType) -> PricingRule | None:
        return self._rules.get((part_type, damage_type))

    async def get_all(self) -> list[PricingRule]:
        return list(self._rules.values())
