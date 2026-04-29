from pydantic_settings import BaseSettings, SettingsConfigDict

from auto_repair_estimator.backend.domain.value_objects.ml_thresholds import (
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
    # Parts: single threshold; default comes from the domain SSOT so
    # the ML worker, backend and tests all agree on the same cutoff.
    # Env PARTS_CONFIDENCE_THRESHOLD overrides without a rebuild.
    parts_confidence_threshold: float = PARTS_CONFIDENCE_THRESHOLD
    # Damages: per-class thresholds live in ml_thresholds.py
    # (DAMAGES_CONFIDENCE_BY_CLASS). This field stays None by default —
    # the detector reads the per-class SSOT directly. If an operator
    # sets env DAMAGES_CONFIDENCE_THRESHOLD to a number, that value
    # acts as a UNIFORM override across ALL damage classes (panic
    # knob for runtime experiments without rebuilding the image).
    damages_confidence_threshold: float | None = None
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
