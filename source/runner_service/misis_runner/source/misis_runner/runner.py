import asyncio
import logging
import time
from uuid import UUID

from misis_runner.app.config import settings
from misis_runner.app.heartbeat_sender import HeartbeatSender
from misis_runner.app.video_processor import VideoProcessor
from misis_runner.inference_client import InferenceClient
from misis_runner.kafka.consumer import Consumer
from misis_runner.s3.s3_client import S3Client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Runner:
    def __init__(self):
        self.active_scenarios: dict[UUID, VideoProcessor] = {}
        self.stopping_scenarios = set()
        self.lock = asyncio.Lock()
        self.consumer = Consumer(self)
        self.heartbeat_sender = HeartbeatSender()
        self.s3_client = S3Client()
        self.inference_client = InferenceClient(
            base_url=settings.INFERENCE_SERVICE_URL
        )
        self.max_concurrent = 1

    async def handle_start(self, scenario_id: UUID, s3_video_key: str, resume_from_frame: int = 0):
        if scenario_id in self.active_scenarios:
            logger.info(f"[Runner] Scenario already runnign {scenario_id}")
            return
        try:
            logger.info(f"[Runner] Starting scenario {scenario_id}")
            processor = VideoProcessor(
                scenario_id=scenario_id,
                s3_video_key=s3_video_key,
                s3_client=self.s3_client,
                inference_client=self.inference_client,
                resume_from_frame=resume_from_frame,
            )
            self.active_scenarios[scenario_id] = processor
            asyncio.create_task(self._run_processor(processor))

        except Exception as e:
            logger.error(f"[Runner] Failed to start scenario {scenario_id}: {str(e)}")
            self.active_scenarios.pop(scenario_id, None)

    async def handle_stop(self, scenario_id: UUID):
        if scenario_id in self.active_scenarios:
            self.stopping_scenarios.add(scenario_id)
            processor = self.active_scenarios.pop(scenario_id)
            if processor:
                processor.stop()

    async def start(self):
        await self.heartbeat_sender.producer.start()
        await self.consumer.start(
            start_handler=self.handle_start,
            stop_handler=self.handle_stop
        )
        await self.heartbeat_sender.start(
            get_active_scenarios=lambda: self.active_scenarios
        )

    def is_available(self):
        return len(self.active_scenarios) < self.max_concurrent

    async def cleanup_scenario(self, scenario_id: UUID):
        async with self.lock:
            self.stopping_scenarios.discard(scenario_id)

    async def stop(self):
        logger.info("[Runner] Stopping runner...")
        await self.heartbeat_sender.stop()
        await self.consumer.stop()
        await self.inference_client.close()
        for processor in self.active_scenarios.values():
            processor.stop()
        self.active_scenarios.clear()
        logger.info("[Runner] Runner stopped")

    async def _run_processor(self, processor: VideoProcessor):
        start_time = time.time()
        try:
            await processor.process()
            logger.info(f"[Runner] Scenario {processor.scenario_id} completed in {time.time() - start_time}s")
        except Exception as e:
            logger.error(f"[Runner] Scenario processing failed: {str(e)}")
        finally:
            if processor.scenario_id in self.active_scenarios:
                del self.active_scenarios[processor.scenario_id]
                logger.info(f"[Runner] Slot released for {processor.scenario_id}")
            try:
                await self.heartbeat_sender.producer.send_heartbeat(
                    processor.scenario_id,
                    processor.last_processed_frame,
                )
            except Exception as e:
                logger.error(f"[Runner] Failed to send final heartbeat for {processor.scenario_id}: {str(e)}")
