import asyncio
import logging

from misis_orchestrator.database.database import get_db_session, engine
from misis_orchestrator.database.base import Base
from misis_orchestrator.health.watchdog import Watchdog
from misis_orchestrator.kafka.consumer import Consumer
from misis_orchestrator.orchestrator_service import OrchestratorService
from misis_orchestrator.outbox.outbox_worker import OutboxWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_task_with_restart(task_func, *args):
    while True:
        try:
            await task_func(*args)
        except Exception as e:
            logger.error(f"Task failed: {e}, restarting in 5 sec")
            await asyncio.sleep(5)


async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = get_db_session
    orchestrator = OrchestratorService(session_factory)
    consumers = Consumer(session_factory)
    watchdog = Watchdog(session_factory, orchestrator)
    outbox_worker = OutboxWorker(session_factory)

    await consumers.start()

    tasks = [
        run_task_with_restart(outbox_worker.start),
        run_task_with_restart(process_heartbeats, consumers, orchestrator),
        run_task_with_restart(watchdog.check_timeouts),
        run_task_with_restart(process_commands, consumers, orchestrator),
        run_task_with_restart(check_inactive_heartbeats, orchestrator),
    ]

    try:
        logger.info("[Orchestrator] Starting orchestrator tasks...")
        await asyncio.gather(*tasks)
    finally:
        await shutdown(consumers, orchestrator, outbox_worker, watchdog)


async def process_heartbeats(consumers: Consumer, orchestrator: OrchestratorService):
    try:
        async for heartbeat in consumers.consume_heartbeats():
            try:
                await orchestrator.update_heartbeat(
                    scenario_id=heartbeat.scenario_id,
                    runner_id=heartbeat.runner_id,
                    last_frame=heartbeat.last_frame
                )
                logger.debug(f"[Orchestrator] Heartbeat received: {heartbeat.scenario_id}")
            except Exception as e:
                logger.error(f"[Orchestrator] Heartbeat update failed: {str(e)}")
    except Exception as e:
        logger.error(f"[Orchestrator] Heartbeat processing failed: {str(e)}")


async def process_commands(consumers: Consumer, orchestrator: OrchestratorService):
    try:
        async for command in consumers.consume_commands():
            try:
                logger.info(f"[Orchestrator] Orchestrator in process command '{command.type}'...")
                await orchestrator.process_command(command)
            except Exception as e:
                logger.error(f"[Orchestrator] Command processing failed: {str(e)}")
    except Exception as e:
        logger.error(f"[Orchestrator] Command processing failed: {str(e)}")


async def check_inactive_heartbeats(orchestrator: OrchestratorService):
    while True:
        try:
            logger.info("[Orchestrator] Check scenarios for inactiveness...")
            await orchestrator.check_inactive_heartbeats()
        except Exception as e:
            logger.error(f"Error in inactive heartbeats check: {e}")
        await asyncio.sleep(15)


async def shutdown(consumers, orchestrator, outbox_worker, watchdog):
    watchdog.running = False
    consumers.running = False
    outbox_worker.running = False

    await consumers.stop()
    await orchestrator.stop()
    await outbox_worker.stop()

    logger.info("[Orchestrator] Orchestrator shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
