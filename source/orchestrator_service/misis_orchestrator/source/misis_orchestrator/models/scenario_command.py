from uuid import UUID

from pydantic import BaseModel


class ScenarioCommand(BaseModel):
    scenario_id: UUID
    type: str
    video_path: str | None = None
    resume_from_frame: int = 0
