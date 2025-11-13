from uuid import UUID
from typing import Dict, Any, List, Optional
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


class SwingFlaw(BaseModel):
    id: str
    title: str
    level: Optional[str] = None
    contact: Optional[str] = None
    ball_shape: Optional[str] = None
    cues: List[str] = []
    drills: List[Dict[str, Any]] = []
    similarity: float  # Cosine similarity score (0-1)


class VideoProcessResponse(BaseModel):
    video_id: UUID
    status: str  # Always "processed" for successful response
    frames: list[FrameResponse]  # All processed frames with local URLs
    metrics: Dict[str, Dict[str, Any]]  # Swing metrics by position (address, top, mid_ds, impact, finish)
    swing_flaws: List[SwingFlaw] = []  # Top 3 most similar swing flaws

