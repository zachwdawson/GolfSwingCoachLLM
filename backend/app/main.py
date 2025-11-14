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

