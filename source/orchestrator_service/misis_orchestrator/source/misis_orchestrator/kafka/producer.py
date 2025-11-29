import json
import logging

from aiokafka import AIOKafkaProducer

from misis_orchestrator.app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Producer:
    def __init__(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
        )

    async def send(self, topic: str, value: dict):
        await self.producer.send(topic, value=value)
        logger.info("[Orchestrator] Producer sent msg")

    async def start(self):
        await self.producer.start()

    async def stop(self):
        await self.producer.stop()
