"""
Swing pattern matcher module.
Finds the most similar swing patterns using pgvector cosine similarity.
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from app.core.config import settings

logger = logging.getLogger(__name__)


def find_similar_swing_patterns(
    swing_vector: List[float],
    limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Find the top N most similar swing patterns using cosine similarity.
    
    Args:
        swing_vector: 16-dimensional vector representing the swing metrics
        limit: Number of top matches to return (default: 3)
    
    Returns:
        List of dictionaries containing:
        - id: Pattern ID
        - title: Pattern title
        - level: Pattern level
        - contact: Contact type
        - ball_shape: Ball shape
        - cues: List of cues
        - drills: List of drills
        - similarity: Cosine similarity score (0-1, higher is more similar)
    """
    if not swing_vector or len(swing_vector) != 16:
        logger.warning(f"Invalid swing vector length: {len(swing_vector) if swing_vector else 0}")
        return []
    
    # Format vector as PostgreSQL array literal
    vec_literal = "[" + ",".join(f"{v:.6f}" for v in swing_vector) + "]"
    
    # Query using pgvector cosine distance operator (<=>)
    # Cosine distance = 1 - cosine similarity
    # We order by distance ascending (most similar first)
    sql = text("""
    SELECT 
        id,
        title,
        level,
        contact,
        ball_shape,
        cues,
        drills,
        1 - (metrics_vector <=> CAST(:swing_vector AS vector)) AS similarity
    FROM swing_patterns
    ORDER BY metrics_vector <=> CAST(:swing_vector AS vector)
    LIMIT :limit
    """.strip())
    
    try:
        engine = create_engine(settings.db_url)
        with engine.connect() as conn:
            result = conn.execute(sql, {
                "swing_vector": vec_literal,
                "limit": limit
            })
            
            patterns = []
            for row in result:
                patterns.append({
                    "id": row.id,
                    "title": row.title,
                    "level": row.level,
                    "contact": row.contact,
                    "ball_shape": row.ball_shape,
                    "cues": row.cues if row.cues else [],
                    "drills": row.drills if row.drills else [],
                    "similarity": float(row.similarity) if row.similarity is not None else 0.0
                })
            
            engine.dispose()
            logger.info(f"Found {len(patterns)} similar swing patterns")
            return patterns
            
    except Exception as e:
        logger.error(f"Error finding similar swing patterns: {e}", exc_info=True)
        return []

