from __future__ import annotations

import asyncio
import json
from typing import Any

from aiokafka import AIOKafkaConsumer
from loguru import logger

from auto_repair_estimator.ml_worker.config import get_config
from auto_repair_estimator.ml_worker.inference.composer import compose
from auto_repair_estimator.ml_worker.inference.cropper import crop_parts
from auto_repair_estimator.ml_worker.inference.damage_detector import DamageDetector
from auto_repair_estimator.ml_worker.inference.parts_detector import PartsDetector
from auto_repair_estimator.ml_worker.inference.preprocessor import preprocess
from auto_repair_estimator.ml_worker.inference.result_publisher import ResultPublisher
from auto_repair_estimator.ml_worker.s3_client import S3Client


async def process_request(
    request_id: str,
    image_key: str,
    s3: S3Client,
    parts_detector: PartsDetector,
    damage_detector: DamageDetector,
    publisher: ResultPublisher,
    config: Any,
) -> None:
    try:
        image_bytes = await s3.download_image(image_key)
        preprocess_result = preprocess(image_bytes, max_bytes=config.max_image_bytes)

        part_detections = parts_detector.predict(preprocess_result.original_image)

        if not part_detections:
            await publisher.publish_error(request_id, "no_parts_detected")
            return

        crops = crop_parts(
            preprocess_result.original_image,
            part_detections,
            request_id,
            config.s3_bucket_crops,
        )

        for crop in crops:
            await s3.upload_image(crop.crop_key, crop.crop_bytes)

        all_damages = []
        for i, crop in enumerate(crops):
            detections = damage_detector.predict(
                crop.crop_bytes, crop.part_type, request_id, i, config.s3_bucket_composites
            )
            all_damages.extend(detections)

        composite_bytes = compose(preprocess_result.original_image, all_damages)
        composited_key = f"{config.s3_bucket_composites}/{request_id}_composite.jpg"
        await s3.upload_image(composited_key, composite_bytes)

        parts_data = [
            {
                "part_type": c.part_type,
                "confidence": c.confidence,
                "bbox": c.bbox,
                "crop_image_key": c.crop_key,
            }
            for c in crops
        ]
        damages_data = [
            {
                "damage_type": d.damage_type,
                "part_type": d.part_type,
                "confidence": d.confidence,
                "mask_image_key": d.mask_key,
            }
            for d in all_damages
        ]

        await publisher.publish_success(
            request_id=request_id,
            parts=parts_data,
            damages=damages_data,
            composited_image_key=composited_key,
        )

    except Exception as exc:
        logger.error("ML pipeline failed for request_id={}: {}", request_id, exc)
        await publisher.publish_error(request_id, str(exc))


async def main() -> None:
    config = get_config()

    s3 = S3Client(config.s3_endpoint, config.s3_access_key, config.s3_secret_key)

    import os

    parts_model_path = config.parts_model_path if os.path.exists(config.parts_model_path) else None
    damages_model_path = config.damages_model_path if os.path.exists(config.damages_model_path) else None

    parts_detector = PartsDetector(
        model_path=parts_model_path or config.parts_model_path,
        confidence_threshold=config.parts_confidence_threshold,
    )
    damage_detector = DamageDetector(
        model_path=damages_model_path or config.damages_model_path,
        confidence_threshold=config.damages_confidence_threshold,
    )

    publisher = ResultPublisher(
        bootstrap_servers=config.kafka_bootstrap_servers,
        topic=config.kafka_topic_inference_results,
    )

    if parts_model_path:
        await asyncio.to_thread(parts_detector.load)
    if damages_model_path:
        await asyncio.to_thread(damage_detector.load)

    await publisher.start()

    consumer = AIOKafkaConsumer(
        config.kafka_topic_inference_requests,
        bootstrap_servers=config.kafka_bootstrap_servers,
        group_id=config.kafka_consumer_group,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode()),
    )
    await consumer.start()
    logger.info("ML Worker started, waiting for inference requests")

    try:
        async for msg in consumer:
            message = msg.value
            request_id = message.get("request_id")
            image_key = message.get("image_key")
            if not request_id or not image_key:
                logger.warning("Invalid inference_request message: {}", message)
                continue
            logger.info("Processing inference request request_id={}", request_id)
            await process_request(
                request_id=str(request_id),
                image_key=str(image_key),
                s3=s3,
                parts_detector=parts_detector,
                damage_detector=damage_detector,
                publisher=publisher,
                config=config,
            )
    except asyncio.CancelledError:
        logger.info("ML Worker cancelled")
    finally:
        await consumer.stop()
        await publisher.stop()


if __name__ == "__main__":
    asyncio.run(main())
