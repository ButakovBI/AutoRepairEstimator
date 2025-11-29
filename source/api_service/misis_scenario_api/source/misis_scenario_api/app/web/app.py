import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from sqlalchemy import text

from misis_scenario_api.app.web.routers import router
from misis_scenario_api.database.database import engine
from misis_scenario_api.kafka.producer import Producer
from misis_scenario_api.outbox.outbox_worker import OutboxWorker
from misis_scenario_api.s3.s3_client import S3Client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] Starting application lifespan")
    app.state.s3_client = S3Client()
    app.state.kafka_producer = Producer()
    await app.state.kafka_producer.start()

    app.state.outbox_worker = OutboxWorker(app.state.kafka_producer)
    worker_task = asyncio.create_task(app.state.outbox_worker.start())

    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: sync_conn.execute(text("SELECT 1")))

    yield

    await app.state.outbox_worker.stop()
    await worker_task
    await app.state.kafka_producer.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MISIS Scenario API",
        lifespan=lifespan
    )
    app.include_router(router)
    return app


app = create_app()
