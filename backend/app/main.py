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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Create frames directory
    os.makedirs(settings.frames_dir, exist_ok=True)
    logger.info(f"Frames directory created/verified: {settings.frames_dir}")
    
    # Initialize database (extensions, tables, swing patterns)
    try:
        initialize_database()
    except Exception as e:
        logger.error(f"Database initialization failed on startup: {e}", exc_info=True)
        # Don't fail startup - app can still function without swing patterns
    
    # Start background worker for frame processing (skip during tests)
    # Note: Queue system is deprecated in favor of synchronous processing
    # Keeping for backward compatibility with manual processing endpoint
    if not os.environ.get("TESTING"):
        start_worker()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}

