"""Verify label consistency — every label dict key is a valid lowercase snake_case identifier."""

from auto_repair_estimator.bot.labels import DAMAGE_LABELS, PART_LABELS


class TestPartLabels:
    def test_all_keys_are_lowercase_snake_case(self):
        for key in PART_LABELS:
            assert key == key.lower()
            assert " " not in key

    def test_contains_at_least_ten_parts(self):
        assert len(PART_LABELS) >= 10


class TestDamageLabels:
    def test_all_keys_are_lowercase_snake_case(self):
        for key in DAMAGE_LABELS:
            assert key == key.lower()
            assert " " not in key

    def test_contains_at_least_four_types(self):
        assert len(DAMAGE_LABELS) >= 4
