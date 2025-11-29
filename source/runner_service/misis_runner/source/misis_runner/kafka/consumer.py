import asyncio
import json
import logging
from uuid import UUID

from aiokafka import AIOKafkaConsumer

from misis_runner.app.config import settings
from misis_runner.models.constants.kafka_topic import KafkaTopic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Consumer:
    def __init__(self, runner):
        self._running = False
        self.runner = runner
        self.start_consumer = AIOKafkaConsumer(
            KafkaTopic.RUNNER_COMMANDS.value,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id="runners_group",
            enable_auto_commit=False,
            max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS,
            session_timeout_ms=60000,
            max_poll_records=1,
        )
        self.stop_consumer = AIOKafkaConsumer(
            KafkaTopic.RUNNER_COMMANDS.value,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=None,
            auto_offset_reset='latest',
            consumer_timeout_ms=10000
        )

    async def start(self, start_handler, stop_handler):
        self._running = True
        await self.start_consumer.start()
        await self.stop_consumer.start()

        asyncio.create_task(self._consume_starts(start_handler))
        asyncio.create_task(self._consume_stops(stop_handler))

    async def _consume_starts(self, handler):
        while self._running:
            try:
                async for msg in self.start_consumer:
                    logger.info("[Runner] msg in start_consumer")
                    if not self._running:
                        logger.info("[Runner] Not running, break...")
                        break

                    data = json.loads(msg.value)
                    logger.info(f"[Runner] Message data: {data}")

                    if not self.runner.is_available():
                        logger.info("Runner busy...")
                        await asyncio.sleep(10)
                        break

                    if UUID(data["scenario_id"]) in self.runner.stopping_scenarios:
                        await self.start_consumer.commit()
                        logger.info("[Runner] Scenario already stopping")
                        continue

                    logger.info(f"[Runner] Consumer mdg data: {data}")

                    if data.get("type") == "start":
                        required_fields = ["scenario_id", "video_path"]
                        if all(field in data for field in required_fields):
                            try:
                                scenario_id = UUID(data["scenario_id"])
                                resume_from_frame = data.get("resume_from_frame", 0)
                                await handler(scenario_id, data["video_path"], resume_from_frame)
                                await self.start_consumer.commit()
                            except Exception as e:
                                logger.error(f"[Runner] Failed to start scenario: {str(e)}")
                        else:
                            missing = [f for f in required_fields if f not in data]
                            logger.warning(f"[Runner] Missing required fields: {missing}")
                            await self.start_consumer.commit()
                    else:
                        logger.info(f"[Runner] Ignoring message of type: {data.get('type')}")
                        await self.start_consumer.commit()
            except Exception as e:
                logger.error(f"[Runner] Start command processing failed: {e}")
                await asyncio.sleep(1)

    async def _consume_stops(self, handler):
        while self._running:
            try:
                async for msg in self.stop_consumer:
                    if not self._running:
                        break
                    data = json.loads(msg.value)
                    logger.info(f"[Runner] Message data: {data}")

                    scenario_id = UUID(data["scenario_id"])

                    if data.get("type") == "stop":
                        logger.info("[Runner] Consumer process stop msg...")
                        async with self.runner.lock:
                            if scenario_id in self.runner.stopping_scenarios:
                                continue
                            self.runner.stopping_scenarios.add(scenario_id)
                        try:
                            await handler(scenario_id)
                        except Exception as e:
                            logger.error(f"Failed to stop scenario: {str(e)}")
                        finally:
                            await self.runner.cleanup_scenario(scenario_id)
            except Exception as e:
                logger.error(f"[Runner] Stop command processing failed: {str(e)}")
                await asyncio.sleep(1)

    async def stop(self):
        self._running = False
        await self.start_consumer.stop()
        await self.stop_consumer.stop()
