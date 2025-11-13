from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import text
from app.models.video import Base


class SwingPattern(Base):
    __tablename__ = "swing_patterns"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    level = Column(String, nullable=True)
    contact = Column(String, nullable=True)
    ball_shape = Column(String, nullable=True)
    metric_expectations = Column(JSONB, nullable=True)
    cues = Column(JSONB, nullable=True)
    drills = Column(JSONB, nullable=True)
    # Use text() for pgvector type - will be handled via raw SQL in migrations/inserts
    metrics_vector = Column(Text, nullable=False)  # Stored as pgvector vector(16)

