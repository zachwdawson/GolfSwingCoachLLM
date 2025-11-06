from uuid import UUID
from pydantic import BaseModel


class VideoCreate(BaseModel):
    video_id: UUID
    s3_key: str


class VideoResponse(BaseModel):
    video_id: UUID
    status: str
    s3_key: str
    frame_urls: list[str] = []

