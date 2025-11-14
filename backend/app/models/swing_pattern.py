from sqlalchemy import Column, String, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import text
from sqlalchemy.types import TypeDecorator
from app.models.video import Base


class JSONBCompat(TypeDecorator):
    """A type that uses JSONB for PostgreSQL and JSON for SQLite."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())


class SwingPattern(Base):
    __tablename__ = "swing_patterns"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    level = Column(String, nullable=True)
    contact = Column(String, nullable=True)
    ball_shape = Column(String, nullable=True)
    metric_expectations = Column(JSONBCompat, nullable=True)
    cues = Column(JSONBCompat, nullable=True)
    drills = Column(JSONBCompat, nullable=True)
    # Use text() for pgvector type - will be handled via raw SQL in migrations/inserts
    metrics_vector = Column(Text, nullable=False)  # Stored as pgvector vector(16)

