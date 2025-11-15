import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload import router as upload_router
from app.core.config import settings
from app.core.db_init import initialize_database
from app.processing.queue import start_worker

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Golf Swing Coach API", version="0.1.0")

# Configure CORS origins - support both localhost and production domains
cors_origins = ["http://localhost:3000"]
# Add production origins from environment variable if set
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    cors_origins.extend([origin.strip() for origin in allowed_origins_env.split(",")])
# Add API base URL as origin if it's different from localhost
if settings.api_base_url and "localhost" not in settings.api_base_url:
    # Extract origin from API base URL (e.g., http://alb-dns-name -> http://alb-dns-name)
    api_origin = settings.api_base_url.rstrip("/")
    if api_origin not in cors_origins:
        cors_origins.append(api_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    import sys
    import asyncio
    
    # Force flush to ensure logs are visible immediately
    sys.stdout.flush()
    sys.stderr.flush()
    
    logger.info("=" * 60)
    logger.info("Starting application startup sequence...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"DB_URL configured: {'Yes' if settings.db_url else 'No'}")
    logger.info(f"DB_URL endpoint: {settings.db_url.split('@')[-1] if '@' in settings.db_url else 'Not set'}")
    
    # Create frames directory
    try:
        os.makedirs(settings.frames_dir, exist_ok=True)
        logger.info(f"✓ Frames directory created/verified: {settings.frames_dir}")
    except Exception as e:
        logger.error(f"✗ Failed to create frames directory: {e}", exc_info=True)
        raise

    import resource
    import psutil

    # Check memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage before DB init: {mem_info.rss / 1024 / 1024:.2f} MB")
    logger.info(f"Memory available: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
    
    # Initialize database in background (non-blocking)
    # This allows the app to start even if DB is slow/unreachable
    logger.info("Starting database initialization in background...")
    try:
        initialize_database()
        logger.info("✓ Database initialization completed successfully")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}", exc_info=True)
        logger.error(f"  Exception type: {type(e).__name__}")
        logger.error(f"  Exception details: {str(e)}")
        # Don't fail startup - app can still function without swing patterns
        logger.warning("Continuing despite database initialization failure...")
    
    # Start background worker for frame processing (skip during tests)
    if not os.environ.get("TESTING"):
        logger.info("Starting background worker...")
        try:
            start_worker()
            logger.info("✓ Background worker started")
        except Exception as e:
            logger.error(f"✗ Failed to start background worker: {e}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("Application startup sequence completed - server ready to accept requests")
    sys.stdout.flush()
    sys.stderr.flush()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}

