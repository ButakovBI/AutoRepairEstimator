from collections.abc import Mapping
from datetime import UTC, datetime

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestStatus


class InMemoryRepairRequestRepository(RepairRequestRepository):
    def __init__(self) -> None:
        self._items: dict[str, RepairRequest] = {}

    async def add(self, request: RepairRequest) -> None:
        if request.idempotency_key is not None:
            # Mirror the Postgres UNIQUE constraint on idempotency_key so
            # tests exercising dedup logic behave identically to production.
            for existing in self._items.values():
                if existing.idempotency_key == request.idempotency_key:
                    raise ValueError(f"duplicate idempotency_key={request.idempotency_key!r}")
        self._items[request.id] = request

    async def get(self, request_id: str) -> RepairRequest | None:
        return self._items.get(request_id)

    async def get_by_idempotency_key(self, idempotency_key: str) -> RepairRequest | None:
        for existing in self._items.values():
            if existing.idempotency_key == idempotency_key:
                return existing
        return None

    async def update(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get_timed_out_requests(self) -> list[RepairRequest]:
        now = datetime.now(UTC)
        terminal = {RequestStatus.DONE, RequestStatus.FAILED}
        return [r for r in self._items.values() if r.status not in terminal and r.timeout_at < now]

    async def get_latest_active_by_chat_id(self, chat_id: int) -> RepairRequest | None:
        terminal = {RequestStatus.DONE, RequestStatus.FAILED}
        # The bot cares about the *most recently started* active session —
        # if a user opens a second session while the first is still idling,
        # the new one represents their current intent. We sort by
        # created_at DESC so the semantics match the Postgres query below.
        active = [r for r in self._items.values() if r.chat_id == chat_id and r.status not in terminal]
        if not active:
            return None
        return max(active, key=lambda r: r.created_at)

    @property
    def items(self) -> Mapping[str, RepairRequest]:
        return self._items
