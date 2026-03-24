from typing import Protocol

from auto_repair_estimator.backend.domain.entities.detected_part import DetectedPart


class PartRepository(Protocol):
    async def add(self, part: DetectedPart) -> None: ...

    async def get_by_request_id(self, request_id: str) -> list[DetectedPart]: ...
