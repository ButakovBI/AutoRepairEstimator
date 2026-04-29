from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "auto_repair"
    db_user: str = "auto_repair"
    db_password: str = "auto_repair"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_raw: str = "raw-images"
    s3_bucket_crops: str = "crops"
    s3_bucket_composites: str = "composites"

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_inference_requests: str = "inference_requests"
    kafka_topic_inference_results: str = "inference_results"
    kafka_topic_notifications: str = "notifications"

    outbox_poll_interval_ms: int = 500
    outbox_batch_size: int = 10
    heartbeat_interval_seconds: int = 30
    request_timeout_minutes: int = 5

    @property
    def db_dsn(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


def get_config() -> BackendConfig:
    return BackendConfig()
