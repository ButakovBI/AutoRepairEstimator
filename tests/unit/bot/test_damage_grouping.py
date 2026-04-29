"""Behavioural tests for ``damage_grouping.group_damages``.

The grouping helper is the single source of truth behind the new
paginated edit keyboard and the group-action handler. Getting its
semantics wrong means either silently losing damages (happened before
when the keyboard factory capped at 8) or acting on stale ids (would
happen if we cached the order and the backend returned a different one).

The invariants under test:

* soft-deleted damages are dropped — no zombies in the UI.
* damages without an ``id`` are dropped — the edit keyboard has no way
  to act on them anyway.
* groups are keyed by ``(part_type, damage_type)`` exactly.
* group order matches first-appearance order (deterministic UX).
* the damage_ids inside a group preserve insertion order.
"""

from __future__ import annotations

from auto_repair_estimator.bot.damage_grouping import DamageGroup, group_damages


class TestGroupDamages:
    def test_empty_input_returns_empty_list(self):
        assert group_damages([]) == []

    def test_identical_pairs_collapse_into_single_group(self):
        damages = [
            {"id": "d1", "part_type": "door", "damage_type": "scratch"},
            {"id": "d2", "part_type": "door", "damage_type": "scratch"},
            {"id": "d3", "part_type": "door", "damage_type": "scratch"},
        ]
        groups = group_damages(damages)
        assert groups == [
            DamageGroup(
                part_type="door",
                damage_type="scratch",
                damage_ids=("d1", "d2", "d3"),
            )
        ]
        assert groups[0].count == 3

    def test_distinct_pairs_become_distinct_groups(self):
        damages = [
            {"id": "d1", "part_type": "door", "damage_type": "scratch"},
            {"id": "d2", "part_type": "hood", "damage_type": "dent"},
        ]
        groups = group_damages(damages)
        # Two buckets, preserving first-appearance order.
        assert [(g.part_type, g.damage_type) for g in groups] == [
            ("door", "scratch"),
            ("hood", "dent"),
        ]

    def test_soft_deleted_damages_are_skipped(self):
        # A freshly-deleted damage may still be in the response payload
        # with ``is_deleted=True``. The UI must not keep showing it.
        damages = [
            {"id": "d1", "part_type": "door", "damage_type": "scratch"},
            {"id": "d2", "part_type": "door", "damage_type": "scratch", "is_deleted": True},
        ]
        groups = group_damages(damages)
        assert len(groups) == 1
        assert groups[0].damage_ids == ("d1",)

    def test_damages_without_id_are_skipped(self):
        # The edit keyboard cannot produce an actionable button without
        # a damage id, so silently dropping them is safer than rendering
        # a button that would hit a 404 on the backend.
        damages = [
            {"part_type": "door", "damage_type": "scratch"},
            {"id": "d1", "part_type": "door", "damage_type": "scratch"},
        ]
        groups = group_damages(damages)
        assert len(groups) == 1
        assert groups[0].damage_ids == ("d1",)

    def test_order_preserves_first_appearance_not_id_order(self):
        # VK re-renders keyboards in the order we send them. Stable order
        # across edit round-trips means the user doesn't see the buttons
        # "jump around" after every action. We key ordering on
        # first-appearance rather than on ids so that 17 scratches added
        # in the order ``d9, d10, d1`` still render as a single group in
        # the position where ``d9`` first appeared.
        damages = [
            {"id": "d9", "part_type": "door", "damage_type": "scratch"},
            {"id": "d1", "part_type": "hood", "damage_type": "dent"},
            {"id": "d10", "part_type": "door", "damage_type": "scratch"},
        ]
        groups = group_damages(damages)
        assert [(g.part_type, g.damage_type) for g in groups] == [
            ("door", "scratch"),
            ("hood", "dent"),
        ]
        assert groups[0].damage_ids == ("d9", "d10")
