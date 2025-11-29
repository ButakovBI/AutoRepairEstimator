from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    S3_ACCESS_KEY: str
    S3_PREDICTIONS_BUCKET: str
    S3_VIDEOS_BUCKET: str
    S3_ENDPOINT: str
    S3_SECRET_KEY: str

    DB_HOST: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_PORT: str = "5432"
    DB_USER: str

    KAFKA_BOOTSTRAP_SERVERS: str
    KAFKA_CONSUMER_TIMEOUT_MS: int = 5000
    KAFKA_MAX_POLL_INTERVAL_MS: int = 300000

    HEARTBEAT_INTERVAL: int = 5  # seconds
    INFERENCE_SERVICE_URL: str = "http://inference_service:8001"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
