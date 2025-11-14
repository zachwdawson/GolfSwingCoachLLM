import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    s3_bucket: str = "golf-coach-videos"
    db_url: str = "postgresql://postgres:postgres@postgres:5432/golfcoach"
    log_level: str = "INFO"
    presigned_url_ttl: int = 3600
    model_checkpoint_path: str = "/app/app/ml/latest_model__epoch_20.pth"
    model_device: str = ""  # Empty means auto-detect (handled in service layer)
    model_seq_len: int = 64
    pose_model_name: str = "movenet_thunder"  # "movenet_lightning" or "movenet_thunder"
    pose_input_size: int = 256  # 192 for lightning, 256 for thunder
    pose_keypoint_threshold: float = 0.11  # Minimum confidence for drawing keypoints
    save_annotated_only: bool = True  # If True, replace original with annotated; if False, save both
    frames_dir: str = "/app/frames"  # Base directory for storing frame images locally
    api_base_url: str = "http://localhost:8000"  # Base URL for API (used for absolute frame URLs)
    openai_api_key: str = ""  # OpenAI API key for practice plan generation

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Also check environment variable directly (for Docker/container environments)
        # This ensures we pick up env vars set by docker-compose even if .env file isn't found
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")


settings = Settings()

