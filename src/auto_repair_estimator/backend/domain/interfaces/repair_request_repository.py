from __future__ import annotations

from typing import Protocol

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest


class RepairRequestRepository(Protocol):
    async def add(self, request: RepairRequest) -> None: ...

    async def get(self, request_id: str) -> RepairRequest | None: ...

    async def get_by_idempotency_key(self, idempotency_key: str) -> RepairRequest | None:
        """Look up an existing request by its VK dedup key (``chat_id:message_id``).

        Used by ``CreateRepairRequestUseCase`` to short-circuit retransmitted
        VK photo events without colliding on the DB UNIQUE constraint.
        """
        ...

    async def update(self, request: RepairRequest) -> None: ...

    async def get_timed_out_requests(self) -> list[RepairRequest]:
        """Return non-terminal requests whose ``timeout_at`` is in the past."""
        ...

    async def get_latest_active_by_chat_id(self, chat_id: int) -> RepairRequest | None:
        """Return the most recent non-terminal request for ``chat_id``.

        "Active" means ``status NOT IN ('done', 'failed')``. The bot uses
        this to decide whether to accept free-text / callback input from a
        user or nudge them to press the "Начать" button. Returns ``None``
        when the user has no active session.
        """
        ...
