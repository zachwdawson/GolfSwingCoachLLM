from uuid import UUID
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.schemas.frame import FrameResponse
from app.schemas.swing_flaw import SwingFlaw


class VideoCreate(BaseModel):
    video_id: UUID
    s3_key: str


class VideoResponse(BaseModel):
    video_id: UUID
    status: str
    s3_key: str
    frame_urls: list[str] = []
    ball_shape: Optional[str] = None
    contact: Optional[str] = None
    description: Optional[str] = None


class VideoProcessResponse(BaseModel):
    video_id: UUID
    status: str  # Always "processed" for successful response
    frames: list[FrameResponse]  # All processed frames with local URLs
    metrics: Dict[str, Dict[str, Any]]  # Swing metrics by position (address, top, mid_ds, impact, finish)
    swing_flaws: List[SwingFlaw] = []  # Top 3 most similar swing flaws
    ball_shape: Optional[str] = None
    contact: Optional[str] = None
    description: Optional[str] = None
    practice_plan: Optional[str] = None  # AI-generated practice plan in markdown format

