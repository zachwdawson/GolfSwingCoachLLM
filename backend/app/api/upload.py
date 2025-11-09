import logging
import uuid
import json
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.models.video import Video, Base
from app.models.frame import Frame  # Import to ensure table is created
from app.schemas.video import VideoCreate, VideoResponse
from app.schemas.frame import FrameResponse, FramesListResponse
from app.services.aws import s3_client
from app.processing.queue import enqueue_frame_extraction

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

    # Enqueue frame extraction
    enqueue_frame_extraction(video_id)
    logger.info(f"Enqueued video for frame extraction: {video_id}")

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

    # Get frame URLs if available
    frames = db.query(Frame).filter(Frame.video_id == video_id).order_by(Frame.index).all()
    frame_urls = []
    for frame in frames:
        url = s3_client.generate_presigned_url(frame.s3_key)
        if url:
            frame_urls.append(url)

    return VideoResponse(
        video_id=video.id,
        status=video.status,
        s3_key=video.s3_key,
        frame_urls=frame_urls,
    )


@router.get("/videos/{video_id}/frames", response_model=FramesListResponse)
async def get_frames(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get all frames for a video with presigned URLs."""
    # Verify video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )

    # Get frames - filter to only return the 4 key positions (Address, Top, Impact, Finish)
    # Event classes: 0=Address, 3=Top, 5=Impact, 7=Finish
    key_event_classes = [0, 3, 5, 7]
    frames = (
        db.query(Frame)
        .filter(Frame.video_id == video_id)
        .filter(Frame.event_class.in_(key_event_classes))
        .order_by(Frame.event_class)
        .all()
    )
    
    frame_responses = []
    for frame in frames:
        url = s3_client.generate_presigned_url(frame.s3_key)
        if url:
            # Parse swing_metrics JSON string to dict
            swing_metrics = None
            if frame.swing_metrics:
                try:
                    swing_metrics = json.loads(frame.swing_metrics)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse swing_metrics for frame {frame.id}")
                    swing_metrics = None
            
            frame_responses.append(
                FrameResponse(
                    frame_id=frame.id,
                    video_id=frame.video_id,
                    index=frame.index,
                    url=url,
                    width=frame.width,
                    height=frame.height,
                    created_at=frame.created_at.isoformat(),
                    event_label=frame.event_label,
                    event_class=frame.event_class,
                    swing_metrics=swing_metrics,
                )
            )

    return FramesListResponse(video_id=video_id, frames=frame_responses)


@router.get("/videos/{video_id}/metrics")
async def get_metrics(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get aggregated swing metrics for a video."""
    # Verify video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )

    # Get frames for the 4 key positions (Address, Top, Impact, Finish)
    key_event_classes = [0, 3, 5, 7]
    position_map = {0: "address", 3: "top", 5: "impact", 7: "finish"}
    
    frames = (
        db.query(Frame)
        .filter(Frame.video_id == video_id)
        .filter(Frame.event_class.in_(key_event_classes))
        .all()
    )
    
    # Aggregate metrics by position
    metrics_by_position = {}
    for frame in frames:
        if frame.event_class in position_map and frame.swing_metrics:
            position_name = position_map[frame.event_class]
            try:
                metrics_dict = json.loads(frame.swing_metrics)
                metrics_by_position[position_name] = metrics_dict
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse swing_metrics for frame {frame.id}")
                metrics_by_position[position_name] = {}
    
    return {"video_id": str(video_id), "metrics": metrics_by_position}


@router.post("/videos/{video_id}/process")
async def process_video(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Manually trigger frame extraction for a video."""
    # Verify video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )

    # Enqueue for processing
    enqueue_frame_extraction(video_id)
    logger.info(f"Manually enqueued video for processing: {video_id}")

    return {"message": f"Video {video_id} queued for processing", "video_id": str(video_id)}

