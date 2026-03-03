from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode


class RepairRequestRepository(Protocol):
    async def add(self, request: RepairRequest) -> None:
        ...

    async def get(self, request_id: str) -> RepairRequest | None:
        ...

    async def update(self, request: RepairRequest) -> None:
        ...

