from uuid import UUID
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from app.schemas.video import SwingFlaw


class FrameResponse(BaseModel):
    frame_id: UUID
    video_id: UUID
    index: int
    url: str  # Presigned URL
    width: int
    height: int
    created_at: str
    event_label: Optional[str] = None  # Event type (e.g., "Address", "Top", etc.)
    event_class: Optional[int] = None  # Event class index (0-7)
    swing_metrics: Optional[Dict[str, Any]] = None  # Parsed swing metrics dict


class FramesListResponse(BaseModel):
    video_id: UUID
    frames: list[FrameResponse]
    swing_flaws: List[SwingFlaw] = []  # Top 3 most similar swing flaws

