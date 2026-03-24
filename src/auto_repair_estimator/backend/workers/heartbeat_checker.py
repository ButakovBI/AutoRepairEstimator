from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger

from auto_repair_estimator.backend.adapters.repositories.postgres_repair_request_repository import (
    PostgresRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.interfaces.outbox_repository import OutboxRepository
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestStatus


class HeartbeatChecker:
    def __init__(
        self,
        request_repository: PostgresRepairRequestRepository,
        outbox_repository: OutboxRepository,
        state_machine: RequestStateMachine,
        notifications_topic: str,
        interval_seconds: int = 30,
    ) -> None:
        self._requests = request_repository
        self._outbox = outbox_repository
        self._sm = state_machine
        self._notifications_topic = notifications_topic
        self._interval = interval_seconds

    async def run(self) -> None:
        logger.info("HeartbeatChecker started (interval={}s)", self._interval)
        while True:
            try:
                await self._check_timeouts()
            except asyncio.CancelledError:
                logger.info("HeartbeatChecker cancelled")
                return
            except Exception as exc:
                logger.error("HeartbeatChecker error: {}", exc)
            await asyncio.sleep(self._interval)

    async def _check_timeouts(self) -> None:
        timed_out = await self._requests.get_timed_out_requests()
        if not timed_out:
            return

        logger.warning("HeartbeatChecker: {} timed-out requests found", len(timed_out))
        for request in timed_out:
            try:
                failed = self._sm.transition(request, RequestStatus.FAILED)
                await self._requests.update(failed)
                event = OutboxEvent(
                    id=str(uuid4()),
                    aggregate_id=request.id,
                    topic=self._notifications_topic,
                    payload={
                        "chat_id": request.chat_id,
                        "request_id": request.id,
                        "type": "request_timeout",
                    },
                    created_at=datetime.now(UTC),
                )
                await self._outbox.add(event)
                logger.info("Request id={} timed out -> FAILED", request.id)
            except Exception as exc:
                logger.error("Failed to process timeout for request id={}: {}", request.id, exc)
