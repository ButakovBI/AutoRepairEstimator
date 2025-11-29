import logging
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from misis_orchestrator.database.tables.scenario import Scenario
from misis_orchestrator.database.tables.heartbeat import Heartbeat
from misis_orchestrator.models.constants.command_type import CommandType
from misis_orchestrator.models.constants.scenario_status import ScenarioStatus
from misis_orchestrator.models.scenario_command import ScenarioCommand
from misis_orchestrator.scenario_db_manager import ScenarioDBManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OrchestratorService:
    def __init__(self, session_factory: AsyncSession):
        self.session_factory = session_factory
        self.scenario_manager = ScenarioDBManager(session_factory)

    async def process_command(self, command: ScenarioCommand):
        if command.type == CommandType.START.value:
            await self._process_start_command(command)
        elif command.type == CommandType.STOP.value:
            await self._process_stop_command(command)

    async def _process_start_command(self, command: ScenarioCommand):
        async for session in self.session_factory():
            try:
                async with session.begin():
                    current_status = await self.scenario_manager.get_status(command.scenario_id, session=session)
                    if current_status not in [ScenarioStatus.INIT_STARTUP.value,
                                              ScenarioStatus.INACTIVE.value]:
                        logger.info(f"[Orch] Ignore start for {command.scenario_id}, status: {current_status}")
                        return

                    await self.scenario_manager.update_status(
                        command.scenario_id,
                        ScenarioStatus.IN_STARTUP_PROCESSING,
                        session=session
                    )
                    logger.info(f"[Orchestrator] Change status to {ScenarioStatus.IN_STARTUP_PROCESSING}")

                    await self.scenario_manager.create_outbox_event(
                        scenario_id=command.scenario_id,
                        event_type=ScenarioStatus.IN_STARTUP_PROCESSING.value,
                        payload={
                            "video_path": command.video_path,
                            "resume_from_frame": command.resume_from_frame
                        },
                        session=session,
                    )
                    logger.info(f"[Orchestrator] Create outbox event with resume_from_frame=={command.resume_from_frame}")

                    await self.scenario_manager.upsert_heartbeat(
                        scenario_id=command.scenario_id,
                        last_frame=command.resume_from_frame,
                        session=session,
                    )
                    logger.info("[Orchestrator] Command start success")
            except Exception as e:
                logger.error(f"[Orchestrator] Error start command: {e}")
                raise

    async def _process_stop_command(self, command: ScenarioCommand):
        async for session in self.session_factory():
            try:
                async with session.begin():
                    last_frame = await session.execute(
                        select(Heartbeat.last_frame)
                        .where(Heartbeat.scenario_id == command.scenario_id)
                    )
                    last_frame = last_frame.scalar_one_or_none() or 0

                    current_status = await self.scenario_manager.get_status(command.scenario_id, session=session)
                    if current_status in [ScenarioStatus.IN_SHUTDOWN_PROCESSING.value, ScenarioStatus.INACTIVE.value]:
                        logger.warning(f"Ignore stop for {command.scenario_id}, status: {current_status}")
                        return
                    await self.scenario_manager.update_status(
                        command.scenario_id,
                        ScenarioStatus.IN_SHUTDOWN_PROCESSING,
                        session=session
                    )
                    logger.info(f"[Orchestrator] Status changed to '{ScenarioStatus.IN_SHUTDOWN_PROCESSING}'")

                    await self.scenario_manager.create_outbox_event(
                        scenario_id=command.scenario_id,
                        event_type=ScenarioStatus.IN_SHUTDOWN_PROCESSING.value,
                        payload={
                            "status": ScenarioStatus.IN_SHUTDOWN_PROCESSING.value
                        },
                        session=session,
                    )
                    logger.info(f"[Orchestrator] Outbox event created for status '{ScenarioStatus.IN_SHUTDOWN_PROCESSING}'")
                    logger.info("[Orchestrator] Command stop success")
            except Exception as e:
                logger.error(f"[Orchestrator] Error stop command: {e}")
                raise

    async def restart_scenario(self, scenario_id: UUID, session: AsyncSession):
        logger.info("[Orchestrator] Restarting scenario...")
        current_status = await self.scenario_manager.get_status(scenario_id, session=session)
        if current_status != ScenarioStatus.ACTIVE.value:
            logger.warning(f"[Orchestrator] Skipping restart for non-active scenario {scenario_id}")
            return
        try:
            video_path = await self.scenario_manager.get_video_path(scenario_id, session=session)
        except NoResultFound:
            logger.error(f"Scenario {scenario_id} not found")
            return
        try:
            await session.execute(
                update(Heartbeat)
                .where(Heartbeat.scenario_id == scenario_id)
                .values(is_active=False)
            )
            logger.info("[Orchestrator] Heartbeat status changed to inactive")

            result = await session.execute(
                select(Heartbeat.last_frame)
                .where(Heartbeat.scenario_id == scenario_id)
            )
            last_frame = result.scalar_one_or_none() or 0
            logger.info(f"[Orchestrator] Get last frame for restarting scenario: '{last_frame}'")

            await self.scenario_manager.update_status(
                scenario_id=scenario_id,
                status=ScenarioStatus.IN_STARTUP_PROCESSING,
                session=session
            )
            logger.info(f"[Orchestrator] Updated status to '{ScenarioStatus.IN_STARTUP_PROCESSING}'")

            await self.scenario_manager.create_outbox_event(
                scenario_id=scenario_id,
                event_type=ScenarioStatus.IN_STARTUP_PROCESSING.value,
                payload={"video_path": video_path,
                         "resume_from_frame": last_frame},
                session=session,
            )
            logger.info(f"[Orchestrator] Created outbox event with '{ScenarioStatus.IN_STARTUP_PROCESSING}' status")

            await self.scenario_manager.upsert_heartbeat(
                scenario_id=scenario_id,
                last_frame=last_frame,
                session=session,
            )
            logger.info("[Orchestrator] Restart scenario success")
        except Exception as e:
            logger.error(f"[Orchestrator] Error restart scenario: {e}")
            raise

    async def update_heartbeat(self, scenario_id: UUID, runner_id: str, last_frame: int):
        try:
            async for session in self.session_factory():
                async with session.begin():
                    if last_frame == -1:
                        logger.info(f"[Orchestrator] Final heartbeat received for {scenario_id}")
                        await self.scenario_manager.update_status(
                            scenario_id,
                            ScenarioStatus.INACTIVE,
                            session=session
                        )
                        await self.scenario_manager.upsert_heartbeat(
                            scenario_id=scenario_id,
                            last_frame=last_frame,
                            session=session,
                            runner_id=runner_id,
                            is_active=False
                        )
                        return

                    query = await session.execute(
                        select(Scenario.status)
                        .where(Scenario.id == scenario_id)
                    )
                    current_status = query.scalar_one_or_none()

                    if current_status == ScenarioStatus.IN_STARTUP_PROCESSING.value:
                        await self.scenario_manager.update_status(
                            scenario_id,
                            ScenarioStatus.ACTIVE,
                            session=session
                        )
                        logger.info(f"[Orchestrator] Status updated to {ScenarioStatus.ACTIVE.value}")
                    elif current_status == ScenarioStatus.IN_SHUTDOWN_PROCESSING.value:
                        await self.scenario_manager.update_status(
                            scenario_id,
                            ScenarioStatus.INACTIVE,
                            session=session
                        )
                        logger.info(f"[Orchestrator] Status updated to {ScenarioStatus.INACTIVE.value}")
                    elif current_status != ScenarioStatus.ACTIVE.value:
                        logger.warning(f"[Orchestrator] Heartbeat rejected for scenario in status {current_status}")
                        return
                    await self.scenario_manager.upsert_heartbeat(
                        scenario_id=scenario_id,
                        last_frame=last_frame,
                        session=session,
                        runner_id=runner_id,
                    )
                    logger.info("[Orchestrator] Update heartbeat success")
        except Exception as e:
            logger.error(f"[Watchdog] Update heartbeat error: {str(e)}")

    async def stop(self):
        await self.producer.stop()

    async def check_inactive_heartbeats(self):
        async for session in self.session_factory():
            async with session.begin():
                query = select(Scenario.id).select_from(Scenario).join(
                    Heartbeat,
                    Scenario.id == Heartbeat.scenario_id
                ).where(
                    Scenario.status == ScenarioStatus.IN_SHUTDOWN_PROCESSING.value,
                    Heartbeat.is_active == False  # noqa E712
                )
                result = await session.execute(query)
                scenario_ids = result.scalars().all()

                for scenario_id in scenario_ids:
                    await self.scenario_manager.update_status(
                        scenario_id,
                        ScenarioStatus.INACTIVE,
                        session=session
                    )
                    logger.info(f"[Orchestrator] Scenario {scenario_id} set to INACTIVE due to inactive heartbeat")
