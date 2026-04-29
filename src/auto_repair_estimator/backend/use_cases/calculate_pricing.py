from __future__ import annotations

from dataclasses import dataclass

from auto_repair_estimator.backend.domain.entities.pricing_result import PricingResult
from auto_repair_estimator.backend.domain.interfaces.damage_repository import DamageRepository
from auto_repair_estimator.backend.domain.services.pricing_service import PricingService


@dataclass
class CalculatePricingInput:
    request_id: str


class CalculatePricingUseCase:
    def __init__(self, damage_repository: DamageRepository, pricing_service: PricingService) -> None:
        self._damages = damage_repository
        self._pricing_service = pricing_service

    async def execute(self, data: CalculatePricingInput) -> PricingResult:
        damages = await self._damages.get_by_request_id(data.request_id)
        return await self._pricing_service.calculate(data.request_id, damages)
