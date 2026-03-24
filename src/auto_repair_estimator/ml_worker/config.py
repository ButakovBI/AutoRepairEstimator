from pydantic_settings import BaseSettings, SettingsConfigDict


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
    parts_confidence_threshold: float = 0.7
    damages_confidence_threshold: float = 0.5
    max_image_bytes: int = 10 * 1024 * 1024


def get_config() -> MLWorkerConfig:
    return MLWorkerConfig()
