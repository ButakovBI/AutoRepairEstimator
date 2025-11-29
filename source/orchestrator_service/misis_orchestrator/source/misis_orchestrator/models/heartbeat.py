from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class HeartbeatModel(BaseModel):
    scenario_id: UUID
    runner_id: str
    last_frame: int
    timestamp: datetime
