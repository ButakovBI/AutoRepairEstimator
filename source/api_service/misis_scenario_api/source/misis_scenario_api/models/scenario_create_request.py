from fastapi import UploadFile
from pydantic import BaseModel


class ScenarioCreateRequest(BaseModel):
    video: UploadFile
