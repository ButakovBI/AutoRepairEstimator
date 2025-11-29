import json
import logging

from aiokafka import AIOKafkaProducer

from misis_scenario_api.app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Producer:
    def __init__(self):
        self.producer = None

    async def send(self, topic: str, value: dict):
        await self.producer.send_and_wait(topic, value=value)
        logger.info(f"[API] Producer sent msg: {value}")

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
        )
        await self.producer.start()

    async def stop(self):
        if self.producer:
            await self.producer.stop()
