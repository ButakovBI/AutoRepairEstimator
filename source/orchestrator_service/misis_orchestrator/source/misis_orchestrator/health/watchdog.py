import asyncio
import logging
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from misis_orchestrator.app.config import settings
from misis_orchestrator.database.tables.heartbeat import Heartbeat
from misis_orchestrator.database.tables.scenario import Scenario
from misis_orchestrator.models.constants.scenario_status import ScenarioStatus
from misis_orchestrator.orchestrator_service import OrchestratorService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Watchdog:
    def __init__(self, session_factory, orchestrator: OrchestratorService):
        self.session_factory = session_factory
        self.orchestrator = orchestrator
        self.running = True

    async def check_timeouts(self):
        while self.running:
            try:
                logger.info("[Watchdog] Checking timeouts...")
                await self._check_timeouts()
                await asyncio.sleep(settings.HEARTBEAT_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"[Watchdog] Check timeouts error: {str(e)}")
                await asyncio.sleep(5)

    async def _check_timeouts(self):
        async for session in self.session_factory():
            async with session.begin():
                try:
                    timed_out_scenarios = await self._get_timed_out_scenarios(session)

                    for scenario_id in timed_out_scenarios:
                        await self._handle_timed_out_scenario(session, scenario_id)
                except Exception as e:
                    logger.error(f"[Watchdog] Checking timeouts error: {str(e)}")
                    await asyncio.sleep(5)

    async def _get_timed_out_scenarios(self, session: AsyncSession):
        timeout = timedelta(seconds=settings.HEARTBEAT_TIMEOUT)
        t = datetime.utcnow() - timeout

        result = await session.execute(
            select(Heartbeat.scenario_id)
            .join(Scenario, Scenario.id == Heartbeat.scenario_id)
            .where(
                Heartbeat.last_timestamp < t,
                Heartbeat.is_active == True,  # noqa E712
                Scenario.status == ScenarioStatus.ACTIVE.value
            )
        )
        return result.scalars().all()

    async def _handle_timed_out_scenario(self, session: AsyncSession, scenario_id: UUID):
        try:
            current_status = await self._get_scenario_status(session, scenario_id)

            if current_status != ScenarioStatus.ACTIVE.value:
                logger.info(f"[Watchdog] Skipping restart for {scenario_id}, status: {current_status}")
                return

            logger.info(f"[Watchdog] Heartbeat timeout for {scenario_id}, restarting...")
            await self.orchestrator.restart_scenario(scenario_id, session)
            logger.info(f"[Watchdog] Scenario {scenario_id} restarted")

        except Exception as e:
            logger.error(f"[Watchdog] Failed to restart scenario {scenario_id}: {str(e)}")

    async def _get_scenario_status(self, session: AsyncSession, scenario_id: UUID) -> str | None:
        result = await session.execute(
            select(Scenario.status)
            .where(Scenario.id == scenario_id)
        )
        return result.scalar_one_or_none()
