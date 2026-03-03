from collections.abc import Mapping
from typing import Dict

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.interfaces.repair_request_repository import RepairRequestRepository


class InMemoryRepairRequestRepository(RepairRequestRepository):
    def __init__(self) -> None:
        self._items: Dict[str, RepairRequest] = {}

    async def add(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get(self, request_id: str) -> RepairRequest | None:
        return self._items.get(request_id)

    async def update(self, request: RepairRequest) -> None:
        if request.id not in self._items:
            self._items[request.id] = request
            return
        self._items[request.id] = request

    @property
    def items(self) -> Mapping[str, RepairRequest]:
        return self._items

