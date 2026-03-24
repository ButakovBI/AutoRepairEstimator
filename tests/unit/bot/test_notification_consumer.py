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

        assert api.messages.send.await_count == 2
        first = api.messages.send.await_args_list[0].kwargs
        assert first["peer_id"] == 456
        assert "вручную" in first["message"].lower()

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
