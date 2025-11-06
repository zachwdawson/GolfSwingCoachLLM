import logging
import uuid
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.models.video import Video, Base
from app.schemas.video import VideoCreate, VideoResponse
from app.services.aws import s3_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Database setup (lazy initialization)
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db_url)
        Base.metadata.create_all(bind=_engine)
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal

ALLOWED_MIME_TYPES = ["video/mp4", "video/quicktime"]
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def get_db():
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/upload", response_model=VideoCreate, status_code=status.HTTP_200_OK)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload video file to S3."""
    logger.info(f"Upload request: {file.filename}")

    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Invalid MIME type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type. Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    # Read and validate size
    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE:
        logger.warning(f"File too large: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    # Generate video ID and S3 key
    video_id = uuid.uuid4()
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "mp4"
    s3_key = f"videos/{video_id}/raw.{file_extension}"

    # Upload to S3
    file_obj = BytesIO(content)
    success = s3_client.upload_file(file_obj, s3_key, file.content_type)

    if not success:
        logger.error(f"Failed to upload to S3: {s3_key}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage",
        )

    # Save to database
    try:
        video = Video(id=video_id, s3_key=s3_key, status="uploaded")
        db.add(video)
        db.commit()
        db.refresh(video)
        logger.info(f"Video uploaded: {video_id}, s3_key: {s3_key}")
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save video record",
        )

    return VideoCreate(video_id=video_id, s3_key=s3_key)


@router.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get video status and signed URLs."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )

    # For now, return empty frame_urls
    frame_urls = []

    return VideoResponse(
        video_id=video.id,
        status=video.status,
        s3_key=video.s3_key,
        frame_urls=frame_urls,
    )

