"""User-facing Russian labels for domain enum values.

Kept in the domain layer so that both the Backend (notes, logs,
unpriceable explanations) and the Bot (inline keyboards, messages)
share a single source of truth. Previously the bot owned these labels
and the backend had no way to render user-visible messages about parts
and damages without a layer-breaking import.

``PART_LABELS`` and ``DAMAGE_LABELS`` are indexed by the string enum
values so callers that receive pre-stringified payloads (HTTP bodies,
Kafka messages) don't need to re-enum them.
"""

from __future__ import annotations

from typing import Final

from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageType,
    PartType,
)

PART_LABELS: Final[dict[str, str]] = {
    PartType.DOOR.value: "Дверь",
    PartType.FRONT_FENDER.value: "Переднее крыло",
    PartType.REAR_FENDER.value: "Заднее крыло",
    PartType.TRUNK.value: "Багажник",
    PartType.HOOD.value: "Капот",
    PartType.ROOF.value: "Крыша",
    PartType.HEADLIGHT.value: "Фара",
    PartType.FRONT_WINDSHIELD.value: "Переднее стекло",
    PartType.REAR_WINDSHIELD.value: "Заднее стекло",
    PartType.SIDE_WINDOW.value: "Боковое стекло",
    PartType.WHEEL.value: "Колесо",
    PartType.BUMPER.value: "Бампер",
}

DAMAGE_LABELS: Final[dict[str, str]] = {
    DamageType.SCRATCH.value: "Царапина",
    DamageType.DENT.value: "Вмятина",
    DamageType.PAINT_CHIP.value: "Скол краски",
    DamageType.RUST.value: "Ржавчина",
    DamageType.CRACK.value: "Трещина",
    DamageType.BROKEN_GLASS.value: "Разбитое стекло",
    DamageType.FLAT_TIRE.value: "Спущенное колесо",
    DamageType.BROKEN_HEADLIGHT.value: "Разбитая фара",
}
