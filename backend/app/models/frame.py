import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from app.models.video import Base


class Frame(Base):
    __tablename__ = "frames"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    index = Column(Integer, nullable=False)  # Frame index (0, 1, 2, ...)
    s3_key = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    event_label = Column(String, nullable=True)  # Event type (e.g., "Address", "Toe-up", etc.)
    event_class = Column(Integer, nullable=True)  # Event class index (0-7)
    pose_keypoints = Column(Text, nullable=True)  # JSON string of pose keypoints [1,1,17,3] format
    swing_metrics = Column(Text, nullable=True)  # JSON string of swing metrics
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

