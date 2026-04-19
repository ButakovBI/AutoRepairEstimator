from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PricingResult:
    """Aggregated repair estimate for a request.

    The totals are kept as ``[min, max]`` ranges to reflect the fact that
    workshop rate cards themselves are ranges (see ``PricingRule``). The
    ``breakdown`` entries carry per-damage min/max values in the same units
    (RUB, hours). ``notes`` accumulates user-facing messages that aren't
    representable as a priced row: currently the scratch-polishing hint and
    the "route wheel issues to a tyre shop" hint.
    """

    request_id: str
    total_hours_min: float
    total_hours_max: float
    total_cost_min: float
    total_cost_max: float
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
