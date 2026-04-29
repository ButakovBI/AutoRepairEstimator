from __future__ import annotations

import asyncio

from loguru import logger

from auto_repair_estimator.backend.adapters.gateways.kafka_consumer import KafkaConsumer
from auto_repair_estimator.backend.use_cases.process_inference_result import (
    InferenceDamageData,
    InferencePartData,
    ProcessInferenceResultInput,
    ProcessInferenceResultUseCase,
)


class InferenceResultConsumer:
    def __init__(self, consumer: KafkaConsumer, use_case: ProcessInferenceResultUseCase) -> None:
        self._consumer = consumer
        self._use_case = use_case

    async def run(self) -> None:
        await self._consumer.start()
        logger.info("InferenceResultConsumer started")
        try:
            async for message in self._consumer.messages():
                try:
                    await self._handle(message)
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.error("Failed to process inference_result message: {}", exc)
        except asyncio.CancelledError:
            logger.info("InferenceResultConsumer cancelled")
        finally:
            await self._consumer.stop()

    async def _handle(self, message: dict) -> None:  # type: ignore[type-arg]
        request_id = message.get("request_id")
        if not request_id:
            logger.warning("inference_result missing request_id, skipping")
            return

        parts = [
            InferencePartData(
                part_type=p["part_type"],
                confidence=p.get("confidence", 0.0),
                bbox=p.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                crop_image_key=p.get("crop_image_key"),
            )
            for p in message.get("parts", [])
        ]
        damages = [
            InferenceDamageData(
                damage_type=d["damage_type"],
                part_type=d["part_type"],
                confidence=d.get("confidence", 0.0),
                mask_image_key=d.get("mask_image_key"),
            )
            for d in message.get("damages", [])
        ]

        data = ProcessInferenceResultInput(
            request_id=str(request_id),
            status=message.get("status", "error"),
            parts=parts,
            damages=damages,
            composited_image_key=message.get("composited_image_key"),
            error_message=message.get("error_message"),
        )
        await self._use_case.execute(data)
        logger.info("Processed inference result for request_id={}", request_id)
