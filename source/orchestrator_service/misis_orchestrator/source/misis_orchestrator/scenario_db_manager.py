import logging
from uuid import UUID

from sqlalchemy import func, insert, select, update

from misis_orchestrator.database.tables.heartbeat import Heartbeat
from misis_orchestrator.database.tables.outbox import Outbox
from misis_orchestrator.database.tables.scenario import Scenario
from misis_orchestrator.models.constants.scenario_status import ScenarioStatus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScenarioDBManager:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create_outbox_event(self, scenario_id: UUID, event_type: str, payload: dict, session):
        await session.execute(
            insert(Outbox).values(
                scenario_id=scenario_id,
                event_type=event_type,
                payload=payload,
                processed=False,
            )
        )

    async def get_video_path(self, scenario_id: UUID, session) -> str:
        result = await session.execute(
            select(Scenario.video_path)
            .where(Scenario.id == scenario_id)
        )
        return result.scalar_one()

    async def get_status(self, scenario_id: UUID, session) -> str:
        result = await session.execute(
            select(Scenario.status)
            .where(Scenario.id == scenario_id)
        )
        return result.scalar_one()

    async def update_status(self, scenario_id: UUID, status: ScenarioStatus, session):
        await session.execute(
            update(Scenario)
            .where(Scenario.id == scenario_id)
            .values(status=status)
        )

    async def upsert_heartbeat(self, scenario_id: UUID, last_frame: int, session, runner_id: str | None = None, is_active: bool | None = True):
        values = {
            "last_timestamp": func.now(),
            "last_frame": last_frame,
            "is_active": is_active
        }

        if runner_id is not None:
            values["runner_id"] = runner_id

        result = await session.execute(
            update(Heartbeat)
            .where(Heartbeat.scenario_id == scenario_id)
            .values(**values)
        )

        if result.rowcount == 0:
            await session.execute(
                insert(Heartbeat)
                .values(scenario_id=scenario_id, **values)
            )
        logger.info(f"[Orchestrator] Update heartbeat with values {values}")
