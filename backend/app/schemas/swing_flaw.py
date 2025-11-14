from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class SwingFlaw(BaseModel):
    id: str
    title: str
    level: Optional[str] = None
    contact: Optional[str] = None
    ball_shape: Optional[str] = None
    cues: List[str] = []
    drills: List[Dict[str, Any]] = []
    similarity: float  # Cosine similarity score (0-1)

