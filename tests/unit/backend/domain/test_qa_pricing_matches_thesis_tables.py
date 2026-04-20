"""QA: PricingService возвращает корректный breakdown по таблицам 5/6.

Эти тесты — исполнимая версия пользовательской таблицы маппинга:

* царапина                → покраска
* ржавчина                → покраска
* вмятина                 → рихтовка + покраска
* отвалившийся кусок      → замена детали
* трещина                 → замена детали
* разбитое стекло / фара  → замена детали
* спущенные шины / колёса → шиномонтаж (не priced row, только note)

Источник цифр — две таблицы из методички (см. изображение в тикете),
те же, что зашиты в ``docker/init.sql`` и ``InMemoryPricingRuleRepository``.
Если эти тесты проходят — значит, для любой комбинации (деталь, тип
повреждения), которую пользователь может выбрать в клавиатурах бота,
он получит ненулевую смету вместо «Кузовной ремонт не требуется».
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_pricing_rule_repository import (
    InMemoryPricingRuleRepository,
)
from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.services.pricing_service import PricingService
from auto_repair_estimator.backend.domain.value_objects.pricing_constants import TYRE_SHOP_NOTE
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
)


def _make_damage(part: PartType, damage: DamageType) -> DetectedDamage:
    return DetectedDamage(
        id=str(uuid4()),
        request_id="req-pricing",
        damage_type=damage,
        part_type=part,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )


# ---------------------------------------------------------------------------
# Тело: кузовные детали × все типы повреждений, которые к ним применимы.
# Кортеж: (part, damage, cost_min, cost_max, hours_min, hours_max).
# Все числа — прямо из таблиц 5 и 6 методички.
# ---------------------------------------------------------------------------

_BODY_EXPECTATIONS: list[tuple[PartType, DamageType, float, float, float, float]] = [
    # hood
    (PartType.HOOD, DamageType.SCRATCH,    10_000,  18_000,  8.0,  8.0),
    (PartType.HOOD, DamageType.RUST,       10_000,  18_000,  8.0,  8.0),
    (PartType.HOOD, DamageType.DENT,       30_000,  35_000, 16.0, 16.0),  # 2 дня
    (PartType.HOOD, DamageType.PAINT_CHIP, 28_000,  30_000, 16.0, 16.0),  # замена 2 дня
    (PartType.HOOD, DamageType.CRACK,      28_000,  30_000, 16.0, 16.0),
    # front fender
    (PartType.FRONT_FENDER, DamageType.SCRATCH,    10_000, 18_000,  8.0,  8.0),
    (PartType.FRONT_FENDER, DamageType.RUST,       10_000, 18_000,  8.0,  8.0),
    (PartType.FRONT_FENDER, DamageType.DENT,       23_000, 30_000, 16.0, 24.0),  # 2-3 дня
    (PartType.FRONT_FENDER, DamageType.PAINT_CHIP, 20_000, 20_000,  8.0,  8.0),  # замена 1 день
    (PartType.FRONT_FENDER, DamageType.CRACK,      20_000, 20_000,  8.0,  8.0),
    # rear fender (replacement 75-100 / 5 дней)
    (PartType.REAR_FENDER, DamageType.SCRATCH,    10_000,  18_000,  8.0,  8.0),
    (PartType.REAR_FENDER, DamageType.RUST,       10_000,  18_000,  8.0,  8.0),
    (PartType.REAR_FENDER, DamageType.DENT,       23_000,  30_000, 16.0, 24.0),
    (PartType.REAR_FENDER, DamageType.PAINT_CHIP, 75_000, 100_000, 40.0, 40.0),
    (PartType.REAR_FENDER, DamageType.CRACK,      75_000, 100_000, 40.0, 40.0),
    # door (replacement 20 / 1.5-2 дня)
    (PartType.DOOR, DamageType.SCRATCH,    10_000, 18_000,  8.0,  8.0),
    (PartType.DOOR, DamageType.RUST,       10_000, 18_000,  8.0,  8.0),
    (PartType.DOOR, DamageType.DENT,       23_000, 30_000, 16.0, 24.0),
    (PartType.DOOR, DamageType.PAINT_CHIP, 20_000, 20_000, 12.0, 16.0),
    (PartType.DOOR, DamageType.CRACK,      20_000, 20_000, 12.0, 16.0),
    # trunk (крышка багажника) — реальный кейс из жалобы пользователя
    (PartType.TRUNK, DamageType.SCRATCH,    10_000, 18_000,  8.0,  8.0),
    (PartType.TRUNK, DamageType.RUST,       10_000, 18_000,  8.0,  8.0),  # <- "ржавчина на багажнике"
    (PartType.TRUNK, DamageType.DENT,       23_000, 30_000, 16.0, 24.0),
    (PartType.TRUNK, DamageType.PAINT_CHIP, 20_000, 20_000, 12.0, 16.0),
    (PartType.TRUNK, DamageType.CRACK,      20_000, 20_000, 12.0, 16.0),
    # roof (replacement 75-100 / 5 дней)
    (PartType.ROOF, DamageType.SCRATCH,    10_000,  18_000,  8.0,  8.0),
    (PartType.ROOF, DamageType.RUST,       10_000,  18_000,  8.0,  8.0),
    (PartType.ROOF, DamageType.DENT,       23_000,  30_000, 16.0, 24.0),
    (PartType.ROOF, DamageType.PAINT_CHIP, 75_000, 100_000, 40.0, 40.0),
    (PartType.ROOF, DamageType.CRACK,      75_000, 100_000, 40.0, 40.0),
    # bumper (вмятина 3-5 / 1-2 дня, замена 18 / 1.5 дня)
    (PartType.BUMPER, DamageType.SCRATCH,    10_000, 18_000,  8.0,  8.0),
    (PartType.BUMPER, DamageType.RUST,       10_000, 18_000,  8.0,  8.0),
    (PartType.BUMPER, DamageType.DENT,        3_000,  5_000,  8.0, 16.0),
    (PartType.BUMPER, DamageType.PAINT_CHIP, 18_000, 18_000, 12.0, 12.0),  # <- "скол краски на бампере"
    (PartType.BUMPER, DamageType.CRACK,      18_000, 18_000, 12.0, 12.0),
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "part,damage,cost_min,cost_max,hours_min,hours_max",
    _BODY_EXPECTATIONS,
    ids=[f"{p.value}-{d.value}" for p, d, *_ in _BODY_EXPECTATIONS],
)
async def test_body_pricing_matches_thesis_rate_card(
    part: PartType,
    damage: DamageType,
    cost_min: float,
    cost_max: float,
    hours_min: float,
    hours_max: float,
) -> None:
    """Для каждой (деталь, повреждение) из таблиц 5/6 сумма и часы должны
    точно совпадать с указанными значениями, и в breakdown должна быть
    ровно одна строка."""

    service = PricingService(_rule_repository=InMemoryPricingRuleRepository())
    result = await service.calculate("req-pricing", [_make_damage(part, damage)])

    assert len(result.breakdown) == 1, (
        f"Ожидалась одна строка breakdown для {part.value}+{damage.value}, "
        f"получено {len(result.breakdown)}: {result.breakdown!r}"
    )
    assert result.total_cost_min == cost_min and result.total_cost_max == cost_max, (
        f"Некорректная цена для {part.value}+{damage.value}: "
        f"ожидалось [{cost_min}..{cost_max}] руб., "
        f"получено [{result.total_cost_min}..{result.total_cost_max}]. "
        "Проверьте таблицу 5 методички."
    )
    assert result.total_hours_min == hours_min and result.total_hours_max == hours_max, (
        f"Некорректные часы для {part.value}+{damage.value}: "
        f"ожидалось [{hours_min}..{hours_max}] ч, "
        f"получено [{result.total_hours_min}..{result.total_hours_max}]. "
        "Проверьте таблицу 6 методички (1 день = 8 ч)."
    )


# ---------------------------------------------------------------------------
# Стёкла и фары: только замена
# ---------------------------------------------------------------------------

_GLASS_EXPECTATIONS = [
    (PartType.FRONT_WINDSHIELD, DamageType.BROKEN_GLASS, 5_000, 10_000, 8.0, 8.0),
    (PartType.REAR_WINDSHIELD,  DamageType.BROKEN_GLASS, 5_000, 10_000, 8.0, 8.0),
    (PartType.SIDE_WINDOW,      DamageType.BROKEN_GLASS, 3_000,  3_000, 8.0, 8.0),
    (PartType.HEADLIGHT, DamageType.BROKEN_HEADLIGHT,    3_000,  3_000, 4.0, 4.0),
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "part,damage,cost_min,cost_max,hours_min,hours_max",
    _GLASS_EXPECTATIONS,
    ids=[f"{p.value}-{d.value}" for p, d, *_ in _GLASS_EXPECTATIONS],
)
async def test_glass_and_headlight_pricing_matches_thesis_rate_card(
    part: PartType,
    damage: DamageType,
    cost_min: float,
    cost_max: float,
    hours_min: float,
    hours_max: float,
) -> None:
    service = PricingService(_rule_repository=InMemoryPricingRuleRepository())
    result = await service.calculate("req-pricing", [_make_damage(part, damage)])

    assert len(result.breakdown) == 1
    assert result.total_cost_min == cost_min and result.total_cost_max == cost_max
    assert result.total_hours_min == hours_min and result.total_hours_max == hours_max


# ---------------------------------------------------------------------------
# Колёса / шины: заявляются шиномонтажом, priced row нет.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    "damage",
    [DamageType.FLAT_TIRE, DamageType.SCRATCH, DamageType.DENT],
    ids=lambda d: d.value,
)
async def test_wheel_damage_routes_to_tyre_shop_without_priced_row(damage: DamageType) -> None:
    """Любое повреждение колеса (по спеке — «спущенные шины и проблемы
    с колёсами») должно уходить в шиномонтаж без строки цены, но с
    заметкой, которую бот покажет пользователю."""

    service = PricingService(_rule_repository=InMemoryPricingRuleRepository())
    result = await service.calculate(
        "req-pricing", [_make_damage(PartType.WHEEL, damage)]
    )

    assert result.breakdown == [], (
        "Колесо не должно попадать в priced breakdown; вместо этого показывается "
        "note о шиномонтаже."
    )
    assert result.total_cost_min == 0.0 and result.total_cost_max == 0.0
    assert TYRE_SHOP_NOTE in result.notes, (
        "Для колёсного повреждения должен быть выставлен TYRE_SHOP_NOTE, "
        "иначе пользователь увидит 'Кузовной ремонт не требуется' без объяснений."
    )


# ---------------------------------------------------------------------------
# Проверяем, что дев-режим backend-а больше не возвращает «всё пусто».
# Это регрессионный guard на конкретную жалобу из тикета.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dev_mode_no_longer_returns_empty_breakdown_for_valid_damages() -> None:
    """Раньше ``_InMemoryPricingRuleRepository`` в ``backend/main.py`` всегда
    возвращал ``None``, поэтому любой valid повреждение выдавал пустую смету
    и сообщение «Кузовной ремонт по этой заявке не требуется»."""

    from auto_repair_estimator.backend.main import _init_dev_state
    from fastapi import FastAPI

    app = FastAPI()
    _init_dev_state(app)
    repo = app.state.pricing_rule_repo

    # Два конкретных кейса из жалобы пользователя:
    rule_rust_trunk = await repo.get_rule(PartType.TRUNK, DamageType.RUST)
    rule_chip_hood = await repo.get_rule(PartType.HOOD, DamageType.PAINT_CHIP)

    assert rule_rust_trunk is not None, (
        "Dev-mode pricing не знает о ржавчине на багажнике — все такие сметы "
        "будут пустыми. Почините InMemoryPricingRuleRepository."
    )
    assert rule_chip_hood is not None, (
        "Dev-mode pricing не знает о сколе краски на капоте — такие сметы "
        "будут пустыми."
    )
    # И цены ненулевые.
    assert rule_rust_trunk.labor_cost_min > 0 and rule_chip_hood.labor_cost_min > 0
