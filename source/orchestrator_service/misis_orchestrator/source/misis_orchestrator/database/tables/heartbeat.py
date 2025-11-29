from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from misis_orchestrator.database.base import Base


class Heartbeat(Base):
    __tablename__ = 'heartbeat'
    scenario_id = Column(UUID, primary_key=True)
    runner_id = Column(String, nullable=True)
    last_timestamp = Column(DateTime, nullable=False)
    last_frame = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)
