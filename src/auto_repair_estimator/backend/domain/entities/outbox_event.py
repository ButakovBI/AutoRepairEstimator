from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class OutboxEvent:
    id: str
    aggregate_id: str
    topic: str
    payload: dict[str, Any]
    created_at: datetime
    published_at: datetime | None = None
