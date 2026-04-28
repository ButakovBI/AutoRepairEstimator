from __future__ import annotations

import asyncio
import io
import json
from typing import Any

from aiokafka import AIOKafkaConsumer
from loguru import logger

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType
from auto_repair_estimator.ml_worker.config import get_config
from auto_repair_estimator.ml_worker.inference.composer import DAMAGE_COLORS, DEFAULT_COLOR, compose
from auto_repair_estimator.ml_worker.inference.cropper import crop_parts
from auto_repair_estimator.ml_worker.inference.damage_detector import DamageDetector
from auto_repair_estimator.ml_worker.inference.mask_artifacts import (
    mask_overlay_png_bytes,
    mask_to_grayscale_png_bytes,
)
from auto_repair_estimator.ml_worker.inference.parts_detector import PartsDetector
from auto_repair_estimator.ml_worker.inference.preprocessor import preprocess
from auto_repair_estimator.ml_worker.inference.result_publisher import ResultPublisher
from auto_repair_estimator.ml_worker.s3_client import S3Client


# Colour used for part-mask overlays. The composer already owns per-
# damage colours in DAMAGE_COLORS; for parts we pick a single neutral
# colour (cyan) since the diagnostic PNG only carries one part at a
# time and we don't want it to clash visually with any damage colour.
_PART_OVERLAY_COLOR: tuple[int, int, int] = (0, 200, 255)
_PART_OVERLAY_ALPHA = 0.45
_DAMAGE_OVERLAY_ALPHA = 0.5


async def _save_part_mask_artifacts(
    s3: S3Client,
    original_image: Any,
    part_detections: list,  # list[PartDetection]; `Any` kept to avoid import cycle noise
    request_id: str,
    bucket: str,
) -> None:
    """Persist one mask PNG + one overlay PNG per detected part.

    Keys:
        <bucket>/<request_id>_part_<i>_<part_type>_mask.png       — grayscale
        <bucket>/<request_id>_part_<i>_<part_type>_overlay.png    — RGB preview

    The overlay is produced against the *full* original image so the
    operator can see where on the frame the mask sits, which is more
    useful than overlaying on the crop (the crop already shows the
    bbox contents; the mask question is "is the shape right?").
    """

    img_w, img_h = original_image.size
    for i, det in enumerate(part_detections):
        if det.mask is None:
            continue
        base_key = f"{bucket}/{request_id}_part_{i}_{det.part_type}"
        try:
            mask_png = mask_to_grayscale_png_bytes(det.mask, target_size=(img_w, img_h))
            await s3.upload_image(f"{base_key}_mask.png", mask_png, content_type="image/png")

            overlay_png = mask_overlay_png_bytes(
                original_image,
                det.mask,
                color=_PART_OVERLAY_COLOR,
                alpha=_PART_OVERLAY_ALPHA,
            )
            await s3.upload_image(f"{base_key}_overlay.png", overlay_png, content_type="image/png")
        except Exception as exc:  # pragma: no cover — diagnostics must never fail the pipeline
            logger.warning(
                "Failed to save part mask artifact for request={} part={} idx={}: {}",
                request_id,
                det.part_type,
                i,
                exc,
            )


async def _save_damage_mask_artifacts(
    s3: S3Client,
    crop_bytes: bytes,
    detections: list,  # list[DamageDetection]
    crop_index: int,
    request_id: str,
    bucket: str,
) -> None:
    """Persist one mask PNG + one overlay PNG per damage detection.

    The mask is saved at the crop's native resolution (that's the
    coordinate space the model emits masks in), and the overlay uses
    the crop as the base image — this answers the question "did the
    model highlight the right region inside this part?" at a glance.
    """

    from PIL import Image as _PILImage

    try:
        crop_image = _PILImage.open(io.BytesIO(crop_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover
        logger.warning("Cannot reopen crop bytes for diagnostics: {}", exc)
        return

    crop_w, crop_h = crop_image.size
    for j, det in enumerate(detections):
        if det.mask is None:
            continue
        base_key = f"{bucket}/{request_id}_damage_{crop_index}_{j}_{det.damage_type}"
        try:
            mask_png = mask_to_grayscale_png_bytes(det.mask, target_size=(crop_w, crop_h))
            await s3.upload_image(f"{base_key}_mask.png", mask_png, content_type="image/png")

            colour = DAMAGE_COLORS.get(det.damage_type, DEFAULT_COLOR)
            overlay_png = mask_overlay_png_bytes(
                crop_image,
                det.mask,
                color=colour,
                alpha=_DAMAGE_OVERLAY_ALPHA,
            )
            await s3.upload_image(f"{base_key}_overlay.png", overlay_png, content_type="image/png")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to save damage mask artifact for request={} crop={} damage={} idx={}: {}",
                request_id,
                crop_index,
                det.damage_type,
                j,
                exc,
            )


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

        part_detections = parts_detector.predict(
            preprocess_result.original_image,
            request_id=request_id,
        )

        if not part_detections:
            await publisher.publish_error(request_id, "no_parts_detected")
            return

        crops = crop_parts(
            preprocess_result.original_image,
            part_detections,
            request_id,
            config.s3_bucket_crops,
            excluded_parts=config.crop_excluded_parts_set,
        )

        for crop in crops:
            await s3.upload_image(crop.crop_key, crop.crop_bytes)

        # Diagnostic artifacts are optional: skipped entirely if the
        # operator disables the flag (reduces MinIO writes for prod).
        if config.save_diagnostic_artifacts:
            await _save_part_mask_artifacts(
                s3=s3,
                original_image=preprocess_result.original_image,
                part_detections=part_detections,
                request_id=request_id,
                bucket=config.s3_bucket_crops,
            )

        all_damages = []
        for i, crop in enumerate(crops):
            detections = damage_detector.predict(
                crop.crop_bytes,
                crop.part_type,
                request_id,
                i,
                config.s3_bucket_composites,
                crop_box_pixels=crop.crop_box_pixels,
            )
            all_damages.extend(detections)

            if config.save_diagnostic_artifacts:
                await _save_damage_mask_artifacts(
                    s3=s3,
                    crop_bytes=crop.crop_bytes,
                    detections=detections,
                    crop_index=i,
                    request_id=request_id,
                    bucket=config.s3_bucket_composites,
                )

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
    # Damages: pass an explicit per-class mapping only when the
    # operator set an env-level uniform override (then ALL classes get
    # that single value). Otherwise pass None so the detector reads
    # the per-class SSOT from ml_thresholds.py — keeping the policy
    # in one place that's easy to audit and edit.
    damage_thresholds: dict[str, float] | None
    if config.damages_confidence_threshold is not None:
        damage_thresholds = {
            dt.value: config.damages_confidence_threshold for dt in DamageType
        }
        logger.warning(
            "DAMAGES_CONFIDENCE_THRESHOLD env override active: applying uniform "
            "cutoff {:.2f} to all damage classes (per-class SSOT bypassed)",
            config.damages_confidence_threshold,
        )
    else:
        damage_thresholds = None
    damage_detector = DamageDetector(
        model_path=damages_model_path or config.damages_model_path,
        thresholds=damage_thresholds,
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
