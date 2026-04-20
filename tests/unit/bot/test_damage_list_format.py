"""Behavior tests for the shared running-list formatter.

The formatter is the single source of truth for how the bot renders
"what's currently in this request" — in both the manual-add
("Добавлено: X — Y\n\n{list}") and the ML-edit header (used above
the edit-damage-type sub-menu). If these two locations ever diverged
the user would see two different numberings or two different orderings
for the same underlying data, which is exactly the confusion the
running-list fix is meant to eliminate.
"""

from __future__ import annotations

from auto_repair_estimator.bot.damage_list_format import format_damage_list


class TestHappyPath:
    def test_renders_numbered_list_with_labels(self) -> None:
        # Part and damage codes are resolved to Russian labels via
        # PART_LABELS / DAMAGE_LABELS so the user never sees raw enum
        # values; the 1-based numbering makes references like "удалить
        # третье повреждение" unambiguous.
        out = format_damage_list(
            [
                {"id": "d1", "part_type": "hood", "damage_type": "scratch"},
                {"id": "d2", "part_type": "door", "damage_type": "dent"},
            ]
        )
        assert "1. Капот — Царапина" in out
        assert "2. Дверь — Вмятина" in out

    def test_uses_custom_header(self) -> None:
        # Different call sites ("Текущий список:", "Обнаруженные
        # повреждения:", "Редактирование повреждений:") share the same
        # body format but pick their own header to match the surrounding
        # prose.
        out = format_damage_list(
            [{"id": "d1", "part_type": "hood", "damage_type": "scratch"}],
            header="Обнаруженные повреждения:",
        )
        assert out.startswith("Обнаруженные повреждения:")


class TestEdgeCases:
    def test_empty_input_returns_empty_placeholder(self) -> None:
        # A header followed by nothing reads like a bug — we emit a
        # dedicated placeholder instead so the user sees a real sentence.
        out = format_damage_list([])
        assert "нет добавленных" in out.lower()

    def test_soft_deleted_entries_are_skipped(self) -> None:
        # The backend returns soft-deleted damages in the list so edit
        # tooling can show history; for user-facing display we hide them
        # so the running list matches what the pricing engine will act on.
        out = format_damage_list(
            [
                {"id": "d1", "part_type": "hood", "damage_type": "scratch", "is_deleted": True},
                {"id": "d2", "part_type": "door", "damage_type": "dent"},
            ]
        )
        assert "Капот" not in out
        assert "Дверь" in out
        # Since one is soft-deleted, the surviving one gets index 1 —
        # NOT 2 — so the user doesn't see gaps like "2. Дверь — Вмятина".
        assert "1. Дверь" in out

    def test_only_soft_deleted_entries_treated_as_empty(self) -> None:
        # If every entry is soft-deleted the effective list is empty,
        # and we must fall through to the placeholder rather than render
        # a header with zero lines.
        out = format_damage_list(
            [{"id": "d1", "part_type": "hood", "damage_type": "scratch", "is_deleted": True}]
        )
        assert "нет добавленных" in out.lower()

    def test_unknown_codes_degrade_to_raw_values(self) -> None:
        # A payload from a future backend version with an unknown
        # part/damage code must NOT crash the bot; we fall back to the
        # raw code so the bug is at least visible to the user.
        out = format_damage_list(
            [{"id": "d1", "part_type": "new_part", "damage_type": "new_damage"}]
        )
        assert "new_part" in out
        assert "new_damage" in out
