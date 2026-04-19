"""FastAPI application entry point.

Infrastructure wiring happens in the lifespan. The app.state object carries all
repository / gateway instances so both production code and tests can inject any
implementation via app.state.<name> without touching module globals.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from auto_repair_estimator.backend.adapters.repositories.in_memory_outbox_repository import InMemoryOutboxRepository
from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
from auto_repair_estimator.backend.api import router as requests_router
from auto_repair_estimator.backend.config import get_config


class _InMemoryDamageRepository:
    """Fallback damage repo for development/testing without Postgres."""

    def __init__(self) -> None:
        self._items: dict[str, object] = {}

    async def add(self, damage: object) -> None:
        from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage

        assert isinstance(damage, DetectedDamage)
        self._items[damage.id] = damage

    async def get_by_request_id(self, request_id: str) -> list[object]:
        from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage

        return [d for d in self._items.values() if isinstance(d, DetectedDamage) and d.request_id == request_id]

    async def get(self, damage_id: str) -> object | None:
        return self._items.get(damage_id)

    async def update(self, damage: object) -> None:
        from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage

        assert isinstance(damage, DetectedDamage)
        self._items[damage.id] = damage

    async def soft_delete(self, damage_id: str) -> None:
        from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage

        if damage_id in self._items:
            d = self._items[damage_id]
            assert isinstance(d, DetectedDamage)
            self._items[damage_id] = DetectedDamage(
                id=d.id,
                request_id=d.request_id,
                damage_type=d.damage_type,
                part_type=d.part_type,
                source=d.source,
                is_deleted=True,
                part_id=d.part_id,
                confidence=d.confidence,
                mask_image_key=d.mask_image_key,
            )


class _InMemoryPricingRuleRepository:
    async def get_rule(self, part_type: object, damage_type: object) -> None:
        return None

    async def get_all(self) -> list[object]:
        return []


def _init_dev_state(app: FastAPI) -> None:
    """Populate app.state with in-memory repos (no infrastructure needed)."""
    config = get_config()
    app.state.request_repo = InMemoryRepairRequestRepository()
    app.state.damage_repo = _InMemoryDamageRepository()
    app.state.pricing_rule_repo = _InMemoryPricingRuleRepository()
    app.state.outbox_repo = InMemoryOutboxRepository()
    app.state.s3_bucket_raw = config.s3_bucket_raw
    app.state.kafka_topic_inference_requests = config.kafka_topic_inference_requests


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = get_config()

    try:
        from auto_repair_estimator.backend.adapters.db.connection import close_pool, create_pool
        from auto_repair_estimator.backend.adapters.gateways.kafka_consumer import KafkaConsumer
        from auto_repair_estimator.backend.adapters.gateways.kafka_producer import KafkaProducer
        from auto_repair_estimator.backend.adapters.gateways.minio_storage_gateway import MinioStorageGateway
        from auto_repair_estimator.backend.adapters.repositories.postgres_damage_repository import (
            PostgresDamageRepository,
        )
        from auto_repair_estimator.backend.adapters.repositories.postgres_outbox_repository import (
            PostgresOutboxRepository,
        )
        from auto_repair_estimator.backend.adapters.repositories.postgres_part_repository import PostgresPartRepository
        from auto_repair_estimator.backend.adapters.repositories.postgres_pricing_rule_repository import (
            PostgresPricingRuleRepository,
        )
        from auto_repair_estimator.backend.adapters.repositories.postgres_repair_request_repository import (
            PostgresRepairRequestRepository,
        )
        from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
        from auto_repair_estimator.backend.use_cases.process_inference_result import ProcessInferenceResultUseCase
        from auto_repair_estimator.backend.workers.heartbeat_checker import HeartbeatChecker
        from auto_repair_estimator.backend.workers.inference_result_consumer import InferenceResultConsumer
        from auto_repair_estimator.backend.workers.outbox_flusher import OutboxFlusher

        pool = await create_pool(config.db_dsn)

        storage = MinioStorageGateway(
            endpoint=config.s3_endpoint,
            access_key=config.s3_access_key,
            secret_key=config.s3_secret_key,
        )
        await storage.ensure_buckets(config.s3_bucket_raw, config.s3_bucket_crops, config.s3_bucket_composites)

        request_repo = PostgresRepairRequestRepository(pool)
        damage_repo = PostgresDamageRepository(pool)
        part_repo = PostgresPartRepository(pool)
        outbox_repo = PostgresOutboxRepository(pool)
        sm = RequestStateMachine()

        # Expose repos via app.state for dependency injection
        app.state.request_repo = request_repo
        app.state.damage_repo = damage_repo
        app.state.pricing_rule_repo = PostgresPricingRuleRepository(pool)
        app.state.outbox_repo = outbox_repo
        app.state.storage = storage
        app.state.s3_bucket_raw = config.s3_bucket_raw
        app.state.kafka_topic_inference_requests = config.kafka_topic_inference_requests

        producer = KafkaProducer(config.kafka_bootstrap_servers)
        await producer.start()

        flusher = OutboxFlusher(
            outbox_repository=outbox_repo,
            kafka_producer=producer,
            poll_interval_ms=config.outbox_poll_interval_ms,
            batch_size=config.outbox_batch_size,
        )
        heartbeat = HeartbeatChecker(
            request_repository=request_repo,
            outbox_repository=outbox_repo,
            state_machine=sm,
            notifications_topic=config.kafka_topic_notifications,
            interval_seconds=config.heartbeat_interval_seconds,
        )

        consumer = KafkaConsumer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            topic=config.kafka_topic_inference_results,
            group_id="backend-inference-results",
        )
        process_uc = ProcessInferenceResultUseCase(
            request_repository=request_repo,
            part_repository=part_repo,
            damage_repository=damage_repo,
            outbox_repository=outbox_repo,
            state_machine=sm,
            notifications_topic=config.kafka_topic_notifications,
        )
        inference_consumer = InferenceResultConsumer(consumer=consumer, use_case=process_uc)

        tasks = [
            asyncio.create_task(flusher.run(), name="outbox-flusher"),
            asyncio.create_task(heartbeat.run(), name="heartbeat-checker"),
            asyncio.create_task(inference_consumer.run(), name="inference-result-consumer"),
        ]
        logger.info("Backend started with Postgres + Kafka + MinIO")

        yield

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await producer.stop()
        await close_pool()

    except Exception as exc:
        logger.warning("Infrastructure unavailable ({}), falling back to dev mode", exc)
        _init_dev_state(app)
        yield


app = FastAPI(title="Auto Repair Estimator Backend", lifespan=lifespan)
app.include_router(requests_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app() -> FastAPI:
    """Create a minimal app with in-memory repos — used in tests that don't need a database."""
    test_app = FastAPI()
    test_app.include_router(requests_router)
    test_app.add_api_route("/health", health)
    _init_dev_state(test_app)
    return test_app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("auto_repair_estimator.backend.main:app", host="0.0.0.0", port=8000, reload=False)
