from __future__ import annotations

from typing import Protocol

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest


class RepairRequestRepository(Protocol):
    async def add(self, request: RepairRequest) -> None: ...

    async def get(self, request_id: str) -> RepairRequest | None: ...

    async def update(self, request: RepairRequest) -> None: ...
