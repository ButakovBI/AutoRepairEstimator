import asyncio
import json
import logging
from uuid import UUID

from aiokafka import AIOKafkaConsumer, ConsumerStoppedError
from sqlalchemy import select

from misis_orchestrator.app.config import settings
from misis_orchestrator.models.constants.command_type import CommandType
from misis_orchestrator.models.constants.kafka_topic import KafkaTopic
from misis_orchestrator.models.constants.scenario_status import ScenarioStatus
from misis_orchestrator.models.heartbeat import HeartbeatModel
from misis_orchestrator.database.tables.heartbeat import Heartbeat
from misis_orchestrator.models.scenario_command import ScenarioCommand

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Consumer:
    def __init__(self, session_factory):
        self.running = True
        self.commands_consumer = AIOKafkaConsumer(
            KafkaTopic.SCENARIO_EVENTS.value,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id="orchestrator_group",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        )
        self.heartbeats_consumer = AIOKafkaConsumer(
            KafkaTopic.HEARTBEATS.value,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id="runners_group",
            value_deserializer=lambda v: json.loads(v.decode())
        )
        self.session_factory = session_factory

    async def start(self):
        await self.commands_consumer.start()
        await self.heartbeats_consumer.start()

    async def consume_commands(self):
        retry_delay = 1
        while self.running:
            try:
                async for msg in self.commands_consumer:
                    if not self.running:
                        break
                    logger.info(f"[Orchestrator] Consumer get msg: {msg.value}")
                    data = msg.value
                    event_type = data.get("event_type")
                    if event_type in [ScenarioStatus.INIT_STARTUP.value, ScenarioStatus.INIT_SHUTDOWN.value]:
                        try:
                            last_frame = await self._get_last_processed_frame(data["scenario_id"])
                            logger.info(f"[Orchestrator] Last processed frame: {last_frame}")
                            command = ScenarioCommand(
                                scenario_id=UUID(data["scenario_id"]),
                                type=CommandType.START.value if event_type == ScenarioStatus.INIT_STARTUP.value else CommandType.STOP.value,
                                video_path=data.get("payload", {}).get("video_path"),
                                resume_from_frame=last_frame,
                            )
                            yield command
                            await self.commands_consumer.commit()
                            retry_delay = 1
                        except Exception as e:
                            logger.error(f"[Orchestrator] Command processing failed: {str(e)}, retrying...")
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 30)
            except ConsumerStoppedError:
                logger.info("[Orchestrator] Consumer was stopped")
                break
            except Exception as e:
                logger.error(f"[Orchestrator] Command consumer error: {str(e)}")
                await asyncio.sleep(1)

    async def consume_heartbeats(self):
        retry_delay = 1
        while self.running:
            try:
                async for msg in self.heartbeats_consumer:
                    if not self.running:
                        break
                    try:
                        logger.info("[Orchestrator] Consumer get heartbeat")
                        data = msg.value
                        command = HeartbeatModel(
                            scenario_id=UUID(data["scenario_id"]),
                            runner_id=data["runner_id"],
                            last_frame=data["last_frame"],
                            timestamp=data["timestamp"]
                        )
                        yield command
                        await self.heartbeats_consumer.commit()
                        retry_delay = 1
                    except Exception as e:
                        logger.error(f"[Orchestrator] Heartbeat processing failed: {str(e)}, retrying...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 30)
            except ConsumerStoppedError:
                break
            except Exception as e:
                logger.error(f"[Orchestrator] Heartbeat consumer error: {str(e)}")
                await asyncio.sleep(1)

    async def stop(self):
        self.running = False
        await self.commands_consumer.stop()
        await self.heartbeats_consumer.stop()

    async def _get_last_processed_frame(self, scenario_id: UUID) -> int:
        async for session in self.session_factory():
            try:
                result = await session.execute(
                    select(Heartbeat.last_frame)
                    .where(Heartbeat.scenario_id == scenario_id)
                    .order_by(Heartbeat.last_timestamp.desc())
                    .limit(1)
                )
                if frame := result.scalar_one_or_none():
                    logger.info(f"[Orch Outbox] Got last frame: {frame}")
                    return frame
                logger.info("[Orch Outbox] Use default frame starting")
                return 0

            except Exception as e:
                logger.error(f"[Orch Outbox] Failed to get last frame for {scenario_id}: {str(e)}")
                return 0
