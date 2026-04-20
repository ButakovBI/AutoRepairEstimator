"""Behavioral tests for VK NotificationConsumer._handle method.

Verifies that each notification type dispatches the correct VK API call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.notification_consumer import NotificationConsumer


def _make_consumer() -> tuple[NotificationConsumer, AsyncMock, AsyncMock]:
    api = AsyncMock()
    api.messages.send = AsyncMock()
    backend = AsyncMock(spec=BackendClient)
    consumer = NotificationConsumer(
        bootstrap_servers="localhost:9092",
        topic="notifications",
        api=api,
        backend=backend,
    )
    return consumer, api, backend


class TestNotificationConsumerHandle:
    async def test_inference_complete_calls_send_inference_result(self):
        consumer, api, backend = _make_consumer()

        with patch(
            "auto_repair_estimator.bot.notification_consumer.send_inference_result",
            new_callable=AsyncMock,
        ) as mock_send:
            await consumer._handle(
                {
                    "chat_id": 123,
                    "request_id": "req-1",
                    "type": "inference_complete",
                    "damages": [{"damage_type": "scratch", "part_type": "hood"}],
                    "composited_image_key": "composites/req-1.jpg",
                }
            )
            mock_send.assert_awaited_once()
            call_kwargs = mock_send.call_args.kwargs
            assert call_kwargs["peer_id"] == 123
            assert call_kwargs["request_id"] == "req-1"
            assert len(call_kwargs["damages"]) == 1

    async def test_inference_failed_sends_manual_fallback_message(self):
        consumer, api, _ = _make_consumer()

        await consumer._handle(
            {
                "chat_id": 456,
                "request_id": "req-2",
                "type": "inference_failed",
            }
        )

        # Three messages: the part-selection pair (intro + keyboard) plus
        # the dedicated "press Начать to retry with another photo" nudge
        # that always carries the start_keyboard on the failure branch.
        assert api.messages.send.await_count == 3
        first = api.messages.send.await_args_list[0].kwargs
        assert first["peer_id"] == 456
        assert "вручную" in first["message"].lower()
        # The last message is the start_keyboard nudge — otherwise a
        # user whose photo failed has no zero-typing way to try again.
        last = api.messages.send.await_args_list[-1].kwargs
        assert "Начать" in last["message"]
        assert last.get("keyboard"), "Failure path must attach start_keyboard"

    async def test_inference_failed_no_parts_detected_reason_is_user_facing(self):
        # The worker emits ``no_parts_detected`` when the model genuinely
        # found no car in the frame; the user-facing copy for that case
        # tells them to send a clearer photo, which is specifically
        # actionable (as opposed to the generic "try again" fallback).
        consumer, api, _ = _make_consumer()

        await consumer._handle(
            {
                "chat_id": 456,
                "request_id": "req-2",
                "type": "inference_failed",
                "error_message": "no_parts_detected",
            }
        )

        first = api.messages.send.await_args_list[0].kwargs
        assert "детал" in first["message"].lower()  # "детали автомобиля"

    async def test_request_timeout_sends_timeout_message(self):
        consumer, api, _ = _make_consumer()

        await consumer._handle(
            {
                "chat_id": 789,
                "request_id": "req-3",
                "type": "request_timeout",
            }
        )

        api.messages.send.assert_awaited_once()
        call_kwargs = api.messages.send.call_args.kwargs
        assert call_kwargs["peer_id"] == 789
        assert "время" in call_kwargs["message"].lower() or "ошибк" in call_kwargs["message"].lower()
        # Always attaches the «Начать» keyboard so the user can resume
        # without having to remember the /start command.
        assert call_kwargs.get("keyboard"), "Timeout notification must include the start keyboard"

    async def test_request_timeout_message_includes_time_when_created_at_provided(self):
        """The creation time must appear in the rendered message so the
        user can disambiguate which of several past requests timed out.
        """

        consumer, api, _ = _make_consumer()

        # Fixed, known ISO-8601 timestamp — matching against the rendered
        # local-time string directly would be fragile (timezones differ
        # across test hosts), so we assert structural properties instead:
        # the message must contain a time-like ``HH:MM`` substring that
        # wasn't in the fallback message.
        await consumer._handle(
            {
                "chat_id": 789,
                "request_id": "req-3",
                "type": "request_timeout",
                "request_created_at": "2026-04-19T14:30:00+00:00",
            }
        )

        msg = api.messages.send.call_args.kwargs["message"]
        # "от DD.MM HH:MM" fragment is the signature of the timestamped variant.
        assert "от " in msg
        # HH:MM — we don't hard-code "14:30" because the rendered time is
        # converted to the test host's local timezone. But *some* two
        # ``HH:MM``-looking tokens must be present.
        import re

        assert re.search(r"\d{2}:\d{2}", msg), f"Expected HH:MM in: {msg!r}"

    async def test_request_timeout_falls_back_cleanly_on_malformed_created_at(self):
        """A broken ``request_created_at`` must not swallow the whole
        notification — the user still needs the "your request failed"
        prompt even if the side-channel timestamp is corrupted.
        """

        consumer, api, _ = _make_consumer()

        await consumer._handle(
            {
                "chat_id": 789,
                "request_id": "req-3",
                "type": "request_timeout",
                "request_created_at": "not-a-timestamp",
            }
        )

        api.messages.send.assert_awaited_once()
        msg = api.messages.send.call_args.kwargs["message"]
        assert "время" in msg.lower() or "ошибк" in msg.lower()

    async def test_missing_chat_id_is_ignored(self):
        consumer, api, _ = _make_consumer()

        await consumer._handle({"request_id": "req-4", "type": "inference_complete"})

        api.messages.send.assert_not_called()

    async def test_unknown_type_is_ignored(self):
        consumer, api, _ = _make_consumer()

        await consumer._handle(
            {
                "chat_id": 100,
                "request_id": "req-5",
                "type": "unknown_event",
            }
        )

        api.messages.send.assert_not_called()
