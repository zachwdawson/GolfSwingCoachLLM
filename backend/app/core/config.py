from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    s3_bucket: str = "golf-coach-videos"
    db_url: str = "postgresql://postgres:postgres@postgres:5432/golfcoach"
    log_level: str = "INFO"
    presigned_url_ttl: int = 3600
    model_checkpoint_path: str = ""
    model_device: str = ""  # Empty means auto-detect (handled in service layer)
    model_seq_len: int = 64

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


settings = Settings()

