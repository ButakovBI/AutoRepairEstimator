from collections.abc import Mapping
from datetime import UTC, datetime

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestStatus


class InMemoryRepairRequestRepository(RepairRequestRepository):
    def __init__(self) -> None:
        self._items: dict[str, RepairRequest] = {}

    async def add(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get(self, request_id: str) -> RepairRequest | None:
        return self._items.get(request_id)

    async def update(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get_timed_out_requests(self) -> list[RepairRequest]:
        now = datetime.now(UTC)
        terminal = {RequestStatus.DONE, RequestStatus.FAILED}
        return [r for r in self._items.values() if r.status not in terminal and r.timeout_at < now]

    @property
    def items(self) -> Mapping[str, RepairRequest]:
        return self._items
