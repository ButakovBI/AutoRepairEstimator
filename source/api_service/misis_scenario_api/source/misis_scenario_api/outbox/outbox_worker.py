import asyncio
import logging
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError

from misis_scenario_api.database.database import get_db_session
from misis_scenario_api.database.tables.outbox import Outbox
from misis_scenario_api.kafka.producer import Producer
from misis_scenario_api.models.constants.kafka_topic import KafkaTopic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OutboxWorker:
    def __init__(self, producer: Producer):
        self.producer = producer
        self.running = False

    async def start(self):
        self.running = True
        while self.running:
            try:
                await self._process_outbox()
                logger.info("[API Outbox] Start process outbox")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"[API Outbox] Outbox worker error: {str(e)}")
                await asyncio.sleep(5)

    async def stop(self):
        self.running = False

    async def _process_outbox(self):
        while self.running:
            async for session in get_db_session():
                try:
                    result = await session.execute(
                        select(Outbox)
                        .where(Outbox.processed == False)  # noqa: E712
                        .order_by(Outbox.created_at)
                        .limit(100)
                        .with_for_update(skip_locked=True)
                    )
                    events = result.scalars().all()

                    if not events:
                        logger.info("[API Outbox] No events")
                        return

                    for event in events:
                        logger.info("[API Outbox] Start process events")
                        try:
                            async with session.begin_nested():
                                await self.producer.send(
                                    KafkaTopic.SCENARIO_EVENTS.value,
                                    {
                                        "event_type": event.event_type,
                                        "scenario_id": str(event.scenario_id),
                                        "payload": event.payload,
                                        "event_id": str(event.id),
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                )

                                await session.execute(
                                    update(Outbox)
                                    .where(Outbox.id == event.id)
                                    .values(processed=True)
                                )
                            logger.info(f"[API Outbox] Processed outbox event {event.id}")

                        except Exception as e:
                            logger.error(f"[API Outbox] Failed to process outbox event {event.id}: {str(e)}")
                    await session.commit()
                except OperationalError as e:
                    logger.warning(f"[API Outbox] Database operation failed: {str(e)}")
                    await session.rollback()
                    raise
