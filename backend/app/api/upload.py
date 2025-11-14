import logging
import uuid
import json
import tempfile
import os
from io import BytesIO
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends, BackgroundTasks, Form
from fastapi.responses import FileResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.models.video import Video, Base
from app.models.frame import Frame  # Import to ensure table is created
from app.schemas.video import VideoCreate, VideoResponse, VideoProcessResponse
from app.schemas.swing_flaw import SwingFlaw
from app.core.swing_vector import build_swing_vector
from app.core.swing_matcher import find_similar_swing_patterns
from app.schemas.frame import FrameResponse, FramesListResponse
from app.services.aws import s3_client
from app.services.openai_service import generate_practice_plan
from app.processing.queue import enqueue_frame_extraction
from app.processing.frames import extract_frames, upload_video_and_frames_to_s3

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


@router.post("/upload", response_model=VideoProcessResponse, status_code=status.HTTP_200_OK)
async def upload_video(
    file: UploadFile = File(...),
    ball_shape: Optional[str] = Form(None),
    contact: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and process video file synchronously. Returns frames and metrics immediately."""
    logger.info(f"Upload request: {file.filename}")
    
    # Map ball_shape values from UI to database format
    ball_shape_mapping = {
        "left": "hook",
        "right": "slice",
        "straight": "normal"
    }
    mapped_ball_shape = None
    if ball_shape:
        mapped_ball_shape = ball_shape_mapping.get(ball_shape.lower(), ball_shape.lower())

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

    # Save video to temporary file for processing
    # This avoids the S3 upload/download roundtrip
    temp_file = None
    temp_file_path = None
    try:
        # Create a temporary file with the video content
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            suffix=f".{file_extension}",
            delete=False
        )
        temp_file_path = temp_file.name
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        temp_file = None  # Close the file handle, but keep the file
        logger.info(f"Saved video to temp file: {temp_file_path}")

        # Save to database (video not yet in S3, will be uploaded after processing)
        try:
            video = Video(
                id=video_id,
                s3_key=s3_key,
                status="uploaded",
                ball_shape=mapped_ball_shape,
                contact=contact.lower() if contact else None,
                description=description
            )
            db.add(video)
            db.commit()
            db.refresh(video)
            logger.info(f"Video record created: {video_id}, s3_key: {s3_key}, ball_shape: {mapped_ball_shape}, contact: {contact}")
        except Exception as e:
            logger.error(f"Database error: {e}")
            db.rollback()
            # Clean up temp file on database error
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save video record",
            )

        # Process video synchronously
        logger.info(f"Starting synchronous video processing: {video_id}")
        result = extract_frames(video_id, db, temp_file_path=temp_file_path)
        
        if result is None:
            # Processing failed
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed",
            )
        
        frames = result["frames"]
        metrics = result["metrics"]
        
        # Identify swing flaws using vector similarity
        swing_flaws = []
        try:
            if metrics:
                # Build swing vector from metrics with user-provided contact and ball_shape
                swing_vector = build_swing_vector(
                    metrics,
                    contact=contact.lower() if contact else "normal",
                    ball_shape=mapped_ball_shape if mapped_ball_shape else "normal"
                )
                # Find top 3 similar swing patterns
                similar_patterns = find_similar_swing_patterns(swing_vector, limit=3)
                # Filter: always include top one, and any others with similarity >= 0.7
                filtered_patterns = []
                for i, pattern in enumerate(similar_patterns):
                    similarity = pattern.get("similarity", 0.0)
                    if i == 0 or similarity >= 0.7:
                        filtered_patterns.append(pattern)
                # Convert to SwingFlaw objects
                swing_flaws = [
                    SwingFlaw(
                        id=pattern["id"],
                        title=pattern["title"],
                        level=pattern.get("level"),
                        contact=pattern.get("contact"),
                        ball_shape=pattern.get("ball_shape"),
                        cues=pattern.get("cues", []),
                        drills=pattern.get("drills", []),
                        similarity=pattern.get("similarity", 0.0)
                    )
                    for pattern in filtered_patterns
                ]
                logger.info(f"Identified {len(swing_flaws)} swing flaws for video {video_id}")
        except Exception as e:
            logger.error(f"Error identifying swing flaws: {e}", exc_info=True)
            # Continue without swing flaws if identification fails
        
        # Generate practice plan using OpenAI
        practice_plan = None
        try:
            if metrics:
                practice_plan = generate_practice_plan(
                    metrics=metrics,
                    swing_flaws=swing_flaws,
                    description=description,
                    ball_shape=mapped_ball_shape,
                    contact=contact.lower() if contact else None
                )
                if practice_plan:
                    logger.info(f"Generated practice plan for video {video_id}")
                else:
                    logger.info(f"Practice plan generation skipped or failed for video {video_id}")
        except Exception as e:
            logger.error(f"Error generating practice plan: {e}", exc_info=True)
            # Continue without practice plan if generation fails
        
        # Add background task for S3 upload
        background_tasks.add_task(
            upload_video_and_frames_to_s3,
            video_id,
            temp_file_path,
            frames
        )
        logger.info(f"Added background task for S3 upload: {video_id}")
        
        # Clean up temp file after processing (S3 upload will use it in background)
        # Note: We keep the file for background upload, but could clean it up if needed
        
        # Convert frames to FrameResponse with absolute URLs
        frame_responses = []
        for frame in frames:
            # Extract filename from s3_key (format: frames/{video_id}/filename.jpg)
            frame_filename = os.path.basename(frame.s3_key)
            # Use absolute URL so frontend can load images correctly
            local_url = f"{settings.api_base_url}/frames/{video_id}/{frame_filename}"
            
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
                    url=local_url,
                    width=frame.width,
                    height=frame.height,
                    created_at=frame.created_at.isoformat(),
                    event_label=frame.event_label,
                    event_class=frame.event_class,
                    swing_metrics=swing_metrics,
                )
            )
        
        logger.info(f"Successfully processed video: {video_id} with {len(frame_responses)} frames")
        
        return VideoProcessResponse(
            video_id=video_id,
            status="processed",
            frames=frame_responses,
            metrics=metrics,
            swing_flaws=swing_flaws,
            ball_shape=mapped_ball_shape,
            contact=contact.lower() if contact else None,
            description=description,
            practice_plan=practice_plan
        )

    except Exception as e:
        logger.error(f"Error saving video to temp file: {e}", exc_info=True)
        # Clean up temp file on error
        if temp_file:
            try:
                temp_file.close()
            except Exception:
                pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save video file",
        )

    return VideoCreate(video_id=video_id, s3_key=s3_key)


@router.get("/frames/{video_id}/{filename}")
async def get_frame_image(
    video_id: uuid.UUID,
    filename: str,
    db: Session = Depends(get_db)
):
    """Serve frame image directly from local filesystem."""
    # Verify video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )
    
    frame_path = os.path.join(settings.frames_dir, str(video_id), filename)
    
    if not os.path.exists(frame_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame not found: {filename}",
        )
    
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"}
    )


@router.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get video status and signed URLs."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )

    # Get frame URLs if available (use absolute URLs)
    frames = db.query(Frame).filter(Frame.video_id == video_id).order_by(Frame.index).all()
    frame_urls = []
    for frame in frames:
        # Extract filename from s3_key (format: frames/{video_id}/filename.jpg)
        frame_filename = os.path.basename(frame.s3_key)
        # Use absolute URL so frontend can load images correctly
        local_url = f"{settings.api_base_url}/frames/{video_id}/{frame_filename}"
        frame_urls.append(local_url)

    return VideoResponse(
        video_id=video.id,
        status=video.status,
        s3_key=video.s3_key,
        frame_urls=frame_urls,
        ball_shape=video.ball_shape,
        contact=video.contact,
        description=video.description,
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

    # Get frames - filter to only return the 5 key positions (Address, Top, Mid-downswing, Impact, Finish)
    # Event classes: 0=Address, 3=Top, 4=Mid-downswing, 5=Impact, 7=Finish
    key_event_classes = [0, 3, 4, 5, 7]
    all_frames = (
        db.query(Frame)
        .filter(Frame.video_id == video_id)
        .filter(Frame.event_class.in_(key_event_classes))
        .order_by(Frame.event_class, Frame.created_at.desc())
        .all()
    )
    
    # Group frames by event_class and take the most recent one for each event
    # This ensures we only return one frame per event class
    frames_by_class = {}
    for frame in all_frames:
        if frame.event_class not in frames_by_class:
            frames_by_class[frame.event_class] = frame
        # If multiple frames exist for the same event_class, keep the most recent one
        elif frame.created_at > frames_by_class[frame.event_class].created_at:
            frames_by_class[frame.event_class] = frame
    
    # Convert to list ordered by event_class
    frames = [frames_by_class[ec] for ec in key_event_classes if ec in frames_by_class]
    
    logger.info(
        f"Returning {len(frames)} frames for video {video_id}: "
        f"{[f.event_label for f in frames]}"
    )
    
    frame_responses = []
    for frame in frames:
        # Extract filename from s3_key (format: frames/{video_id}/filename.jpg)
        frame_filename = os.path.basename(frame.s3_key)
        # Use absolute URL so frontend can load images correctly
        local_url = f"{settings.api_base_url}/frames/{video_id}/{frame_filename}"
        
        # Parse swing_metrics JSON string to dict
        swing_metrics = None
        if frame.swing_metrics:
            try:
                swing_metrics = json.loads(frame.swing_metrics)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse swing_metrics for frame {frame.id}")
                swing_metrics = None
        
        logger.debug(
            f"Frame {frame.id}: event_class={frame.event_class}, "
            f"event_label={frame.event_label}, s3_key={frame.s3_key}"
        )
        
        frame_responses.append(
            FrameResponse(
                frame_id=frame.id,
                video_id=frame.video_id,
                index=frame.index,
                url=local_url,
                width=frame.width,
                height=frame.height,
                created_at=frame.created_at.isoformat(),
                event_label=frame.event_label,
                event_class=frame.event_class,
                swing_metrics=swing_metrics,
            )
        )

    # Compute swing flaws from metrics
    swing_flaws = []
    try:
        # Aggregate metrics by position from frames
        metrics_by_position = {}
        position_map = {0: "address", 3: "top", 4: "mid_ds", 5: "impact", 7: "finish"}
        for frame in frames:
            if frame.event_class in position_map and frame.swing_metrics:
                position_name = position_map[frame.event_class]
                try:
                    metrics_dict = json.loads(frame.swing_metrics)
                    metrics_by_position[position_name] = metrics_dict
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse swing_metrics for frame {frame.id}")
        
        if metrics_by_position:
            # Build swing vector from metrics using stored contact and ball_shape values
            swing_vector = build_swing_vector(
                metrics_by_position,
                contact=video.contact.lower() if video.contact else "normal",
                ball_shape=video.ball_shape if video.ball_shape else "normal"
            )
            # Find top 3 similar swing patterns
            similar_patterns = find_similar_swing_patterns(swing_vector, limit=3)
            # Filter: always include top one, and any others with similarity >= 0.7
            filtered_patterns = []
            for i, pattern in enumerate(similar_patterns):
                similarity = pattern.get("similarity", 0.0)
                if i == 0 or similarity >= 0.7:
                    filtered_patterns.append(pattern)
            # Convert to SwingFlaw objects
            swing_flaws = [
                SwingFlaw(
                    id=pattern["id"],
                    title=pattern["title"],
                    level=pattern.get("level"),
                    contact=pattern.get("contact"),
                    ball_shape=pattern.get("ball_shape"),
                    cues=pattern.get("cues", []),
                    drills=pattern.get("drills", []),
                    similarity=pattern.get("similarity", 0.0)
                )
                for pattern in filtered_patterns
            ]
            logger.info(f"Computed {len(swing_flaws)} swing flaws for video {video_id}")
    except Exception as e:
        logger.error(f"Error computing swing flaws in get_frames: {e}", exc_info=True)
        # Continue without swing flaws if computation fails

    return FramesListResponse(video_id=video_id, frames=frame_responses, swing_flaws=swing_flaws)


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

    # Get frames for the 5 key positions (Address, Top, Mid-downswing, Impact, Finish)
    key_event_classes = [0, 3, 4, 5, 7]
    position_map = {0: "address", 3: "top", 4: "mid_ds", 5: "impact", 7: "finish"}
    
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

