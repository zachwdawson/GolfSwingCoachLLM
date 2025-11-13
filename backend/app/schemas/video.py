from uuid import UUID
from typing import Dict, Any
from pydantic import BaseModel
from app.schemas.frame import FrameResponse


class VideoCreate(BaseModel):
    video_id: UUID
    s3_key: str


class VideoResponse(BaseModel):
    video_id: UUID
    status: str
    s3_key: str
    frame_urls: list[str] = []


class VideoProcessResponse(BaseModel):
    video_id: UUID
    status: str  # Always "processed" for successful response
    frames: list[FrameResponse]  # All processed frames with local URLs
    metrics: Dict[str, Dict[str, Any]]  # Swing metrics by position (address, top, mid_ds, impact, finish)

