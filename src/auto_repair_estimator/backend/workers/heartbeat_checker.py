from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger

from auto_repair_estimator.backend.domain.entities.outbox_event import OutboxEvent
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.outbox_repository import OutboxRepository
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


class HeartbeatChecker:
    def __init__(
        self,
        request_repository: RepairRequestRepository,
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
                logger.info("Request id={} timed out -> FAILED", request.id)

                if await self._should_notify_user(request):
                    await self._outbox.add(self._build_timeout_event(request))
                else:
                    logger.debug(
                        "Suppressing request_timeout notification for id={} "
                        "(mode={}, suppressed by active-sibling rule)",
                        request.id,
                        request.mode.value,
                    )
            except Exception as exc:
                logger.error("Failed to process timeout for request id={}: {}", request.id, exc)

    async def _should_notify_user(self, failed_request: RepairRequest) -> bool:
        """Decide whether to push a user-facing timeout message for ``failed_request``.

        Business rules (from user guide / product requirements):

        * Only ML requests time out in a way the user needs to know about —
          MANUAL requests don't involve asynchronous processing, so a watchdog
          sweep that catches a stale MANUAL session would only confuse the user.
        * If the user already has another active (non-terminal) request in
          ANY mode for the same chat, they are clearly mid-flow; shouting
          "your old request timed out, press /start" on top of their live
          session is noise, so we swallow the notification.
        """

        if failed_request.mode is not RequestMode.ML:
            return False

        # At this point ``failed_request`` itself was just updated to FAILED,
        # so ``get_latest_active_by_chat_id`` will never return it. Any
        # non-None result is therefore a genuinely *other* active request.
        sibling = await self._requests.get_latest_active_by_chat_id(failed_request.chat_id)
        return sibling is None

    def _build_timeout_event(self, request: RepairRequest) -> OutboxEvent:
        """Construct the outbox payload for a user-visible timeout notification.

        ``request_created_at`` is included so the bot can render "your request
        from HH:MM ..." and the user can disambiguate which of several past
        sessions timed out. We emit ISO-8601 UTC — the bot formats for display.
        """

        return OutboxEvent(
            id=str(uuid4()),
            aggregate_id=request.id,
            topic=self._notifications_topic,
            payload={
                "chat_id": request.chat_id,
                "request_id": request.id,
                "type": "request_timeout",
                "request_created_at": request.created_at.isoformat(),
            },
            created_at=datetime.now(UTC),
        )
