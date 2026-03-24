from pydantic_settings import BaseSettings, SettingsConfigDict


class BotConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vk_group_token: str = "changeme"
    vk_group_id: int = 0
    backend_url: str = "http://localhost:8000"

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_notifications: str = "notifications"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"


def get_config() -> BotConfig:
    return BotConfig()
