from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_HOST: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_PORT: str = "5432"
    DB_USER: str

    KAFKA_BOOTSTRAP_SERVERS: str

    HEARTBEAT_TIMEOUT: int = 25  # seconds
    HEARTBEAT_CHECK_INTERVAL: int = 5  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
