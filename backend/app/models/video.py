import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    s3_key = Column(String, nullable=False)
    status = Column(String, nullable=False, default="uploaded")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

