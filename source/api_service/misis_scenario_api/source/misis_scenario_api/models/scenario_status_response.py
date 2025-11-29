from uuid import UUID

from pydantic import BaseModel


class ScenarioStatusResponse(BaseModel):
    scenario_id: UUID
    status: str
