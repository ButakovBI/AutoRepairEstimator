from pydantic_settings import BaseSettings, SettingsConfigDict

from auto_repair_estimator.backend.domain.value_objects.ml_thresholds import (
    DAMAGES_CONFIDENCE_THRESHOLD,
    PARTS_CONFIDENCE_THRESHOLD,
)


class MLWorkerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_inference_requests: str = "inference_requests"
    kafka_topic_inference_results: str = "inference_results"
    kafka_consumer_group: str = "ml-worker"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_raw: str = "raw-images"
    s3_bucket_crops: str = "crops"
    s3_bucket_composites: str = "composites"

    parts_model_path: str = "/app/models/parts.pt"
    damages_model_path: str = "/app/models/damages.pt"
    # Defaults come from the domain SSOT so the ML worker, backend and
    # tests all agree on the same business-level cutoffs. Operators can
    # still override via env (PARTS_CONFIDENCE_THRESHOLD /
    # DAMAGES_CONFIDENCE_THRESHOLD) without a rebuild.
    parts_confidence_threshold: float = PARTS_CONFIDENCE_THRESHOLD
    damages_confidence_threshold: float = DAMAGES_CONFIDENCE_THRESHOLD
    max_image_bytes: int = 10 * 1024 * 1024

    # Comma-separated list of PartType values to skip during cropping.
    # Example: "headlight" — set via env CROP_EXCLUDED_PARTS.
    # Empty by default: damage detection runs on every detected part.
    crop_excluded_parts: str = ""

    # When True, every intermediate artifact (part masks, damage masks,
    # per-damage overlays on the crop) is persisted to MinIO next to
    # the crops/composites. These files are only read by humans (via
    # the MinIO console), so the flag gives operators a cheap switch
    # between "developer view" and "production" storage footprints.
    # Defaults to True so the current dev/stage workflow gets full
    # diagnostic visibility out of the box.
    save_diagnostic_artifacts: bool = True

    @property
    def crop_excluded_parts_set(self) -> frozenset[str]:
        return frozenset(p.strip() for p in self.crop_excluded_parts.split(",") if p.strip())


def get_config() -> MLWorkerConfig:
    return MLWorkerConfig()
