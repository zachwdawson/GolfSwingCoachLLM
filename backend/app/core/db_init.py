"""
Database initialization module.
Handles setup of database extensions, tables, and initial data.
"""
import json
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from app.core.config import settings

logger = logging.getLogger(__name__)


def enable_pgvector_extension(engine):
    """Enable pgvector extension if not already enabled."""
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")
    except Exception as e:
        logger.error(f"Failed to enable pgvector extension: {e}")
        raise


def migrate_videos_table(engine):
    """Add new columns to videos table if they don't exist (migration)."""
    try:
        with engine.begin() as conn:
            # Check if columns exist and add them if they don't
            # Check for ball_shape column
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'videos' AND column_name = 'ball_shape'
            """))
            if result.fetchone() is None:
                conn.execute(text("ALTER TABLE videos ADD COLUMN ball_shape VARCHAR"))
                logger.info("Added ball_shape column to videos table")
            
            # Check for contact column
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'videos' AND column_name = 'contact'
            """))
            if result.fetchone() is None:
                conn.execute(text("ALTER TABLE videos ADD COLUMN contact VARCHAR"))
                logger.info("Added contact column to videos table")
            
            # Check for description column
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'videos' AND column_name = 'description'
            """))
            if result.fetchone() is None:
                conn.execute(text("ALTER TABLE videos ADD COLUMN description TEXT"))
                logger.info("Added description column to videos table")
            
            logger.info("Videos table migration complete")
    except Exception as e:
        logger.error(f"Failed to migrate videos table: {e}")
        raise


def create_swing_patterns_table(engine):
    """Create swing_patterns table with pgvector column if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS swing_patterns (
        id VARCHAR PRIMARY KEY,
        title VARCHAR NOT NULL,
        level VARCHAR,
        contact VARCHAR,
        ball_shape VARCHAR,
        metric_expectations JSONB,
        cues JSONB,
        drills JSONB,
        metrics_vector vector(16) NOT NULL
    );
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
            logger.info("swing_patterns table created or already exists")
    except Exception as e:
        logger.error(f"Failed to create swing_patterns table: {e}")
        raise


def initialize_swing_patterns(engine):
    """
    Initialize swing patterns from swing_flaws.json.
    This function is idempotent - safe to call multiple times.
    """
    # Import here to avoid circular imports
    from app.metric_distribution.build_vector_db import build_metrics_vector
    
    # Get the path to swing_flaws.json relative to the build_vector_db.py location
    script_dir = Path(__file__).parent.parent / "metric_distribution"
    json_path = script_dir / "swing_flaws.json"
    
    if not json_path.exists():
        logger.warning(f"swing_flaws.json not found at {json_path}, skipping swing patterns initialization")
        return
    
    logger.info(f"Loading swing flaws from {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            flaws = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load swing_flaws.json: {e}")
        return
    
    inserted_count = 0
    skipped_count = 0
    
    for entry in flaws:
        vec = build_metrics_vector(entry)
        
        # Prepare VALUES
        row_id = entry["id"]
        title = entry.get("title", "")
        level = entry.get("level", "")
        contact = entry.get("contact", "")
        ball_shape = entry.get("ball-shape", "")
        
        metric_expectations_json = json.dumps(entry.get("metric_expectations", {}),
                                              ensure_ascii=False)
        cues_json = json.dumps(entry.get("cues", []), ensure_ascii=False)
        drills_json = json.dumps(entry.get("drills", []), ensure_ascii=False)
        
        # pgvector literal: '[v1, v2, ...]'
        vec_literal = "[" + ",".join(f"{v:.3f}" for v in vec) + "]"
        
        sql = text("""
INSERT INTO swing_patterns (
  id, title, level, contact, ball_shape,
  metric_expectations, cues, drills, metrics_vector
) VALUES (
  :id,
  :title,
  :level,
  :contact,
  :ball_shape,
  CAST(:metric_expectations AS JSONB),
  CAST(:cues AS JSONB),
  CAST(:drills AS JSONB),
  CAST(:metrics_vector AS vector)
)
ON CONFLICT (id) DO NOTHING
""".strip())
        
        try:
            with engine.begin() as conn:
                result = conn.execute(sql, {
                    "id": row_id,
                    "title": title,
                    "level": level,
                    "contact": contact,
                    "ball_shape": ball_shape,
                    "metric_expectations": metric_expectations_json,
                    "cues": cues_json,
                    "drills": drills_json,
                    "metrics_vector": vec_literal
                })
                if result.rowcount > 0:
                    inserted_count += 1
                    logger.debug(f"Inserted swing pattern: {row_id} - {title}")
                else:
                    skipped_count += 1
        except IntegrityError:
            skipped_count += 1
            logger.debug(f"Skipped duplicate swing pattern: {row_id} - {title}")
        except Exception as e:
            logger.error(f"Failed to insert swing pattern {row_id}: {e}")
            # Continue with other entries even if one fails
            continue
    
    logger.info(f"Swing patterns initialization complete: {inserted_count} inserted, {skipped_count} skipped")


def initialize_database():
    """
    Initialize database: enable extensions, create tables, and load initial data.
    This function is idempotent and safe to call multiple times.
    """
    try:
        engine = create_engine(settings.db_url)
        
        # Enable pgvector extension
        enable_pgvector_extension(engine)
        
        # Migrate videos table (add new columns if needed)
        migrate_videos_table(engine)
        
        # Create swing_patterns table
        create_swing_patterns_table(engine)
        
        # Initialize swing patterns data
        initialize_swing_patterns(engine)
        
        engine.dispose()
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        # Don't raise - allow app to start even if initialization fails
        # The app can still function, just without swing patterns

