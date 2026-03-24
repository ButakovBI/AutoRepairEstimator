from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PricingResult:
    request_id: str
    total_hours: float
    total_cost: float
    breakdown: list[dict[str, Any]] = field(default_factory=list)
