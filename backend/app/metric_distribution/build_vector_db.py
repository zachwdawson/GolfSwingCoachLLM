import json
import logging
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Config / mappings
# -----------------------------

# Order of dimensions in the metrics_vector:
VECTOR_FIELDS = [
    "contact_normal",
    "contact_fat",
    "contact_thin",
    "contact_inconsistent",
    "ball_shape_normal",
    "ball_shape_hook",
    "ball_shape_slice",
    "address_spine_forward_bend_deg",
    "address_shoulder_alignment_deg",
    "top_shoulder_turn_deg",
    "top_pelvis_turn_deg",
    "mid_ds_hip_open_deg",
    "mid_ds_trail_elbow_flexion_deg",
    "impact_hip_open_deg",
    "impact_forward_shaft_lean_deg",
    "finish_balance_over_lead_foot_norm",
]

# Indices (for convenience)
IDX = {name: i for i, name in enumerate(VECTOR_FIELDS)}

# Contact and ball-shape categories
CONTACT_CATS = ["normal", "fat", "thin", "inconsistent"]
BALL_SHAPE_CATS = ["normal", "hook", "slice"]

# Map expectation string -> z-score
def expectation_to_z(exp: str) -> float:
    """
    exp is one of "normal", "high", "low".
    For high/low: ±1.5 as requested. Normal -> 0.
    """
    if exp is None:
        return 0.0
    exp = exp.lower()
    if exp == "high":
        return 1.5
    if exp == "low":
        return -1.5
    # Treat anything else ("normal", unknown) as 0
    return 0.0


# -----------------------------
# Core conversion
# -----------------------------

def build_metrics_vector(entry: dict) -> list[float]:
    """
    Given a flaw entry with:
      - "contact": "normal" | "fat" | "thin" | "inconsistent"
      - "ball-shape": "normal" | "hook" | "slice"
      - "metric_expectations": {metric_name: "normal"|"high"|"low"}
    build a 16-dim vector as specified.
    """
    vec = [0.0] * len(VECTOR_FIELDS)

    # 1) One-hot encode contact
    contact = entry.get("contact", "normal").lower()
    if contact in CONTACT_CATS:
        vec[IDX[f"contact_{contact}"]] = 1.0

    # 2) One-hot encode ball shape
    ball_shape = entry.get("ball-shape", "normal").lower()
    if ball_shape in BALL_SHAPE_CATS:
        vec[IDX[f"ball_shape_{ball_shape}"]] = 1.0

    # 3) Metric expectations -> z-scores
    me = entry.get("metric_expectations", {})

    # helper to fill metric dimension if present
    def set_metric(metric_key: str, vec_name: str):
        exp = me.get(metric_key)  # "high" / "low" / "normal"
        z = expectation_to_z(exp)
        vec[IDX[vec_name]] = z

    set_metric("address_spine_forward_bend_deg",
               "address_spine_forward_bend_deg")
    set_metric("address_shoulder_alignment_deg",
               "address_shoulder_alignment_deg")
    set_metric("top_shoulder_turn_deg",
               "top_shoulder_turn_deg")
    set_metric("top_pelvis_turn_deg",
               "top_pelvis_turn_deg")
    set_metric("mid_ds_hip_open_deg",
               "mid_ds_hip_open_deg")
    set_metric("mid_ds_trail_elbow_flexion_deg",
               "mid_ds_trail_elbow_flexion_deg")
    set_metric("impact_hip_open_deg",
               "impact_hip_open_deg")
    set_metric("impact_forward_shaft_lean_deg",
               "impact_forward_shaft_lean_deg")
    set_metric("finish_balance_over_lead_foot_norm",
               "finish_balance_over_lead_foot_norm")

    return vec


# -----------------------------
# Database setup and operations
# -----------------------------

def enable_pgvector_extension(engine):
    """Enable pgvector extension if not already enabled."""
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")
    except Exception as e:
        logger.error(f"Failed to enable pgvector extension: {e}")
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


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    json_path = script_dir / "swing_flaws.json"
    
    if not json_path.exists():
        logger.error(f"swing_flaws.json not found at {json_path}")
        sys.exit(1)

    # Load your JSON (the array you posted, with metric_expectations added)
    logger.info(f"Loading swing flaws from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        flaws = json.load(f)

    # Connect to database
    logger.info(f"Connecting to database: {settings.db_url.split('@')[-1] if '@' in settings.db_url else '***'}")
    engine = create_engine(settings.db_url)

    try:
        # Enable pgvector extension
        enable_pgvector_extension(engine)
        
        # Create table
        create_swing_patterns_table(engine)

        # Insert data
        inserted_count = 0
        skipped_count = 0
        
        for entry in flaws:
            vec = build_metrics_vector(entry)

            # Prepare VALUES
            row_id = entry["id"]
            title = entry.get("title", "")
            level = entry.get("level", "")
            contact = entry.get("contact", "")
            # Column in DB is ball_shape (underscore), JSON uses "ball-shape"
            ball_shape = entry.get("ball-shape", "")

            metric_expectations_json = json.dumps(entry.get("metric_expectations", {}),
                                                  ensure_ascii=False)
            cues_json = json.dumps(entry.get("cues", []), ensure_ascii=False)
            drills_json = json.dumps(entry.get("drills", []), ensure_ascii=False)

            # pgvector literal: '[v1, v2, ...]' - format as string for PostgreSQL
            vec_literal = "[" + ",".join(f"{v:.3f}" for v in vec) + "]"

            # Use parameterized query for safety, but cast vector literal directly
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
  :metric_expectations::jsonb,
  :cues::jsonb,
  :drills::jsonb,
  :metrics_vector::vector
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
                    # Check if row was actually inserted (ON CONFLICT returns 0 rows affected if skipped)
                    if result.rowcount > 0:
                        inserted_count += 1
                        logger.info(f"Inserted swing pattern: {row_id} - {title}")
                    else:
                        skipped_count += 1
                        logger.warning(f"Skipped duplicate swing pattern: {row_id} - {title}")
            except IntegrityError:
                skipped_count += 1
                logger.warning(f"Skipped duplicate swing pattern (IntegrityError): {row_id} - {title}")
            except Exception as e:
                logger.error(f"Failed to insert swing pattern {row_id}: {e}")
                raise

        logger.info(f"Insertion complete: {inserted_count} inserted, {skipped_count} skipped")
        
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
