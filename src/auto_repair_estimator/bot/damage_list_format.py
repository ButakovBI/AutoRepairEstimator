"""Shared formatter for the running "what's currently in this request" list.

Before this module existed the manual-add flow and the ML-edit flow each
had their own half-rendered version of the same list: manual showed only
the last-added damage ("Добавлено: Капот — Царапина") with no overview,
ML-edit embedded the list in button labels but hid it the moment the
user stepped into the edit-type sub-menu. The result was the user
forgetting which damages had already been added or which they were
about to revise. One formatter, used in every place the user reasons
about "what's in this basket", keeps the wording identical.

Accepts the loose dict shape that the backend ``/v1/requests/{id}``
response returns — ``part_type``, ``damage_type``, ``is_deleted``. Soft-
deleted entries are filtered out so an edit round-trip doesn't keep
showing crossed-out damages.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


def format_damage_list(
    damages: Iterable[dict[str, Any]],
    *,
    header: str = "Текущий список повреждений:",
    empty_text: str = "Пока нет добавленных повреждений.",
) -> str:
    """Render ``damages`` as a human-readable, numbered list.

    Skips soft-deleted entries (``is_deleted=True``). When the resulting
    list is empty we emit ``empty_text`` instead of a header followed by
    nothing — a lone header reads like a bug.
    """

    active = [d for d in damages if not d.get("is_deleted", False)]
    if not active:
        return empty_text

    lines: list[str] = [header]
    for index, damage in enumerate(active, 1):
        part_label = PART_LABELS.get(damage.get("part_type", ""), damage.get("part_type", "?"))
        damage_label = DAMAGE_LABELS.get(damage.get("damage_type", ""), damage.get("damage_type", "?"))
        lines.append(f"{index}. {part_label} — {damage_label}")
    return "\n".join(lines)
