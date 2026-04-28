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
                # Race guard: between the batch fetch and this iteration the
                # row may have been moved to a terminal state by another
                # actor (most commonly ``AbandonRequestUseCase`` triggered
                # by the user pressing «Начать» or sending a replacement
                # photo). The in-memory ``request`` snapshot still says
                # PROCESSING/etc., so without a reload we would (a)
                # overwrite ``ml_error_code="user_abandoned"`` with the
                # None carried by the stale snapshot and (b) emit a
                # user-facing timeout notification for a request the user
                # already closed. Re-read the live row and skip if the
                # transition is no longer applicable.
                fresh = await self._requests.get(request.id)
                if fresh is None:
                    logger.debug(
                        "Request id={} vanished between batch fetch and timeout"
                        " processing; skipping",
                        request.id,
                    )
                    continue
                if fresh.status in {RequestStatus.DONE, RequestStatus.FAILED}:
                    logger.info(
                        "Request id={} already terminal ({}) by the time the"
                        " heartbeat reached it (likely user_abandoned); skipping",
                        request.id,
                        fresh.status.value,
                    )
                    continue

                failed = self._sm.transition(fresh, RequestStatus.FAILED)
                await self._requests.update(failed)
                logger.info("Request id={} timed out -> FAILED", request.id)

                if await self._should_notify_user(fresh):
                    await self._outbox.add(self._build_timeout_event(fresh))
                else:
                    logger.debug(
                        "Suppressing request_timeout notification for id={} "
                        "(mode={}, suppressed by active-sibling rule)",
                        request.id,
                        fresh.mode.value,
                    )
            except Exception as exc:
                logger.error("Failed to process timeout for request id={}: {}", request.id, exc)

    # Statuses where "the model is working on it and the user is just
    # waiting" — CREATED (photo being uploaded), QUEUED (sitting in
    # Kafka), PROCESSING (ML worker running). Timing out during any of
    # these is something the user legitimately needs to know about.
    # PRICING means inference already succeeded and the user is editing
    # the damage list; a watchdog timeout here is a backend state-machine
    # concern, not a user-visible "try again" situation, so we stay
    # silent to avoid interrupting the edit session.
    _NOTIFIABLE_PRE_TIMEOUT_STATUSES = frozenset(
        {RequestStatus.CREATED, RequestStatus.QUEUED, RequestStatus.PROCESSING}
    )

    async def _should_notify_user(self, pre_timeout_request: RepairRequest) -> bool:
        """Decide whether to push a user-facing timeout message.

        ``pre_timeout_request`` is the snapshot as it was just *before* the
        watchdog flipped it to FAILED — its ``status`` field therefore
        tells us what the user was actually doing when the timeout fired.

        Business rules (from user guide / product requirements):

        * Only ML requests time out in a way the user needs to know about —
          MANUAL requests don't involve asynchronous processing, so a watchdog
          sweep that catches a stale MANUAL session would only confuse the user.
        * Only pre-PRICING ML statuses trigger the user notification: those
          are the states where the user is waiting on the model. If the
          request was already in PRICING (user was editing after
          inference), a "timeout, try again" popup is useless — the
          inference already succeeded and the user was mid-UI.
        * If the user already has another active (non-terminal) request in
          ANY mode for the same chat, they are clearly mid-flow; shouting
          "your old request timed out, press /start" on top of their live
          session is noise, so we swallow the notification.
        """

        if pre_timeout_request.mode is not RequestMode.ML:
            return False

        if pre_timeout_request.status not in self._NOTIFIABLE_PRE_TIMEOUT_STATUSES:
            return False

        # At this point the request itself has just been updated to FAILED,
        # so ``get_latest_active_by_chat_id`` will never return it. Any
        # non-None result is therefore a genuinely *other* active request.
        sibling = await self._requests.get_latest_active_by_chat_id(pre_timeout_request.chat_id)
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
