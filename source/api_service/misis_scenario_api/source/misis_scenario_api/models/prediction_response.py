from uuid import UUID

from pydantic import BaseModel

from misis_scenario_api.models.bounding_box import BoundingBox


class PredictionResponse(BaseModel):
    scenario_id: UUID
    predictions: list[BoundingBox]
