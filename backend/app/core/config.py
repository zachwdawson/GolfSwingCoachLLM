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
    pose_model_name: str = "movenet_thunder"  # "movenet_lightning" or "movenet_thunder"
    pose_input_size: int = 256  # 192 for lightning, 256 for thunder
    pose_keypoint_threshold: float = 0.11  # Minimum confidence for drawing keypoints
    save_annotated_only: bool = True  # If True, replace original with annotated; if False, save both

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


settings = Settings()

