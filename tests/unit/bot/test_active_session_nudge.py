"""Behavior tests for ``active_session_nudge(mode, status)``.

The helper centralises the off-script reply the bot sends when a user
types text while already inside a scenario. Before it existed the bot
always said "use the buttons in the last message", which was false for
the ML/CREATED state (the "send me a photo" prompt has no keyboard
attached). Every test below pins one concrete (mode, status) pair to a
user-observable property — the exact wording is intentionally not
regexed so copy tweaks don't break the test, but the *shape* of the
nudge must hold.
"""

from __future__ import annotations

import pytest

from auto_repair_estimator.bot.handlers.start import (
    active_session_nudge,
)


class TestMLCreatedBranch:
    def test_prompts_user_to_send_photo(self) -> None:
        # ML + CREATED is the "send me a photo" state — the previous
        # message has no buttons, so the nudge must explicitly ask for a
        # photo rather than referring to non-existent buttons.
        msg = active_session_nudge("ml", "created")
        assert "фотограф" in msg.lower()


class TestMLProcessingBranch:
    @pytest.mark.parametrize("status", ["queued", "processing"])
    def test_tells_user_to_wait_for_ml_result(self, status: str) -> None:
        # While the ML model is running the only correct answer is "wait
        # for the result" — the user has no pending action.
        msg = active_session_nudge("ml", status)
        assert "обрабатыв" in msg.lower()


class TestManualWaitingBranch:
    def test_points_at_buttons_in_last_message(self) -> None:
        # In manual mode the last message always has buttons (part list
        # or damage-type list), so the generic "use the buttons" nudge
        # is actually correct here — and only here.
        msg = active_session_nudge("manual", "pricing")
        assert "кнопк" in msg.lower()


class TestPricingBranch:
    @pytest.mark.parametrize("mode", ["ml", "manual"])
    def test_points_at_buttons_for_both_modes(self, mode: str) -> None:
        # Once the request has reached PRICING the last bot message
        # always carries an action keyboard (add_more_or_confirm or the
        # ML inference-result keyboard), so the generic "use the buttons"
        # copy is correct regardless of the original mode.
        msg = active_session_nudge(mode, "pricing")
        assert "кнопк" in msg.lower()


class TestUnknownInputsDegradeGracefully:
    @pytest.mark.parametrize(
        "mode, status",
        [
            (None, None),
            ("", ""),
            ("unknown_mode", "pricing"),
            ("ml", "some_future_status"),
        ],
    )
    def test_never_raises_and_always_mentions_the_start_button(
        self, mode: str | None, status: str | None
    ) -> None:
        # Unknown / missing inputs must produce a non-empty nudge
        # mentioning the universal escape hatch so users always see a
        # way forward even if the backend contract drifts.
        msg = active_session_nudge(mode, status)
        assert "Начать" in msg


class TestAllBranchesMentionStartButton:
    @pytest.mark.parametrize(
        "mode, status",
        [
            ("ml", "created"),
            ("ml", "queued"),
            ("ml", "processing"),
            ("ml", "pricing"),
            ("manual", "pricing"),
        ],
    )
    def test_every_nudge_tells_user_how_to_reset(self, mode: str, status: str) -> None:
        # Regardless of which branch we fall into the user must always
        # see "нажмите «Начать»" in the text — otherwise they are stuck
        # if the contextual hint doesn't match what they want to do.
        msg = active_session_nudge(mode, status)
        assert "Начать" in msg
