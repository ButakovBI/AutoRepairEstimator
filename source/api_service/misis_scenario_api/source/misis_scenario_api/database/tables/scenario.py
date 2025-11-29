from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID

from misis_scenario_api.database.base import Base


class Scenario(Base):
    __tablename__ = "scenario"

    id = Column(UUID(as_uuid=True), primary_key=True)
    status = Column(String(50), nullable=False)
    video_path = Column(String(255), nullable=False)
