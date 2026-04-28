"""Group active damages by ``(part_type, damage_type)`` for the edit UI.

Historically the edit keyboard rendered one button per raw damage. With
the new model that routinely detects 13 scratches on a single door, the
8-button cap in ``damage_edit_keyboard`` started silently dropping the
tail — users could not edit damages #9+ at all.

This helper collapses the flat damage list into stable groups. The
grouping is deterministic (sorted by first appearance) so VK re-renders
the same button order between edits. Groups keep the list of underlying
``damage_id`` values so the handler can bulk-apply edits / deletes.

Soft-deleted damages (``is_deleted=True``) are dropped before grouping,
matching the rest of the bot's contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DamageGroup:
    """Collapsed view of all damages sharing ``(part_type, damage_type)``.

    ``damage_ids`` preserves the ingestion order so "Удалить одно" removes
    the oldest item first — the user's mental model is "the first one I
    see is the one I'm complaining about".
    """

    part_type: str
    damage_type: str
    damage_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def count(self) -> int:
        return len(self.damage_ids)


def group_damages(damages: Iterable[dict[str, Any]]) -> list[DamageGroup]:
    """Collapse ``damages`` into groups sorted by first appearance.

    Non-deleted damages without a resolvable id are skipped — they cannot
    be acted upon by the edit keyboard anyway, and pulling them into a
    group would make bulk operations silently drop them.
    """

    order: list[tuple[str, str]] = []
    buckets: dict[tuple[str, str], list[str]] = {}
    for damage in damages:
        if damage.get("is_deleted", False):
            continue
        damage_id = damage.get("id")
        if not damage_id:
            continue
        part_type = str(damage.get("part_type", ""))
        damage_type = str(damage.get("damage_type", ""))
        key = (part_type, damage_type)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(str(damage_id))

    return [DamageGroup(part_type=pt, damage_type=dt, damage_ids=tuple(buckets[(pt, dt)])) for pt, dt in order]
