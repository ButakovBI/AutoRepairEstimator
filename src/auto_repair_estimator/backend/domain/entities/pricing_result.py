from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PricingResult:
    request_id: str
    total_hours: float
    total_cost: float
    breakdown: dict[str, Any]

