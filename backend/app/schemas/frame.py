from uuid import UUID
from pydantic import BaseModel


class FrameResponse(BaseModel):
    frame_id: UUID
    video_id: UUID
    index: int
    url: str  # Presigned URL
    width: int
    height: int
    created_at: str


class FramesListResponse(BaseModel):
    video_id: UUID
    frames: list[FrameResponse]

