import logging
import subprocess
import tempfile
import os
import time
import json
import shutil
import numpy as np
from typing import Optional, Dict, Any, List
from uuid import UUID
from PIL import Image
from sqlalchemy.orm import Session
from app.models.video import Video
from app.models.frame import Frame
from app.services.aws import s3_client
from app.core.config import settings
from app.ml.service import get_model
from app.ml.preprocessing import preprocess_video, prepare_video_for_inference
from app.ml.inference import process_video_for_events, EVENT_NAMES
from app.processing.pose_estimation import estimate_pose, draw_pose_overlay
from app.processing.swing_metrics import compute_metrics

logger = logging.getLogger(__name__)

# Number of event frames to extract (8 golf swing events)
NUM_EVENT_FRAMES = 8


def extract_frames(video_id: UUID, db: Session, temp_file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Extract event frames from a video using model-based labeling.
    Saves frames locally and writes metadata to database.
    
    Args:
        video_id: UUID of the video to process
        db: Database session
        temp_file_path: Optional path to temporary file containing the video.
                       If provided, uses this file instead of downloading from S3.
    
    Returns:
        Dict containing:
            - frames: list[Frame] - All processed frames
            - metrics: Dict[str, Dict[str, Any]] - Swing metrics by position
        None on failure
    """
    temp_video_path = None
    temp_file_owned = False  # Track if we created the temp file (need to clean up)
    
    try:
        # Get video record
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video not found: {video_id}")
            return None

        # Update video status
        video.status = "processing"
        db.commit()

        # Load model
        model = get_model()
        if model is None:
            logger.error("Model not available, cannot extract frames")
            video.status = "failed"
            db.commit()
            return None

        # Determine video file path: use temp_file_path if provided, otherwise download from S3
        if temp_file_path:
            logger.info(f"Validating temp file: {temp_file_path}")
            
            # Step 1: Check if file exists
            if not os.path.exists(temp_file_path):
                logger.warning(f"Temp file does not exist: {temp_file_path}, falling back to S3 download")
                temp_file_path = None
            else:
                # Step 2: Check file size
                try:
                    file_size = os.path.getsize(temp_file_path)
                    if file_size == 0:
                        logger.warning(f"Temp file is empty (0 bytes): {temp_file_path}, falling back to S3 download")
                        temp_file_path = None
                    else:
                        # Step 3: Check file is readable
                        try:
                            with open(temp_file_path, "rb") as f:
                                f.read(1)  # Try reading first byte
                            # All validations passed
                            temp_video_path = temp_file_path
                            temp_file_owned = False  # Don't delete the file, it's managed by upload endpoint
                            logger.info(f"Temp file validated successfully: {temp_file_path} (size: {file_size} bytes)")
                        except (IOError, OSError) as e:
                            logger.warning(f"Temp file is not readable: {temp_file_path}, error: {e}, falling back to S3 download")
                            temp_file_path = None
                except OSError as e:
                    logger.warning(f"Failed to get temp file size: {temp_file_path}, error: {e}, falling back to S3 download")
                    temp_file_path = None
        
        if not temp_video_path:
            # Fallback: Download video from S3 to temp file
            logger.info(f"Downloading video from S3: {video.s3_key}")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video_path = temp_video.name
                temp_file_owned = True  # We created this, need to clean up
                try:
                    # Download from S3
                    if not s3_client.download_file(video.s3_key, temp_video_path):
                        logger.error(f"Failed to download video from S3: {video.s3_key}")
                        video.status = "failed"
                        db.commit()
                        return None
                    logger.info(f"Downloaded video from S3: {video.s3_key}")
                except Exception as e:
                    logger.error(f"Error downloading from S3: {e}", exc_info=True)
                    video.status = "failed"
                    db.commit()
                    return None

        try:
            # Create local frames directory for this video
            local_frames_dir = os.path.join(settings.frames_dir, str(video_id))
            os.makedirs(local_frames_dir, exist_ok=True)
            logger.info(f"Created frames directory: {local_frames_dir}")

            # Get video FPS for timestamp calculation
            fps = get_video_fps(temp_video_path)
            if fps is None:
                logger.error(f"Failed to get video FPS: {video_id}")
                video.status = "failed"
                db.commit()
                return None

            # Preprocess video for model
            logger.info("Preprocessing video for model inference")
            preprocessed_video = preprocess_video(temp_video_path)

            # Prepare video for inference
            device = next(model.parameters()).device
            video_tensor, n_sequences = prepare_video_for_inference(
                preprocessed_video, settings.model_seq_len, str(device)
            )

            # Run model inference to get event frames
            logger.info("Running model inference to identify event frames")
            start_time = time.time()
            event_frames = process_video_for_events(
                model, video_tensor, n_sequences, settings.model_seq_len
            )
            inference_time = time.time() - start_time
            logger.info(f"Model inference completed in {inference_time:.2f}s")

            if not event_frames:
                logger.error("No event frames identified by model")
                video.status = "failed"
                db.commit()
                return None

            # Log the predicted frame indices for debugging
            logger.info("Model predicted event frames:")
            frame_idx_to_events = {}  # Track which events map to the same frame_idx
            for event_class, frame_idx in sorted(event_frames.items()):
                event_label = EVENT_NAMES.get(event_class, f"Event_{event_class}")
                logger.info(f"  {event_label} (class {event_class}): frame_idx={frame_idx}")
                if frame_idx not in frame_idx_to_events:
                    frame_idx_to_events[frame_idx] = []
                frame_idx_to_events[frame_idx].append((event_class, event_label))
            
            # Check for duplicate frame indices
            duplicates = {idx: events for idx, events in frame_idx_to_events.items() if len(events) > 1}
            if duplicates:
                logger.warning(
                    f"Multiple events mapped to the same frame indices: {duplicates}. "
                    f"This may cause incorrect frame labeling."
                )

            # Extract frames using ffmpeg at predicted indices
            logger.info(f"Extracting {len(event_frames)} event frames")
            with tempfile.TemporaryDirectory() as temp_dir:
                # Dictionary to store keypoints for metrics computation
                # Maps position name to keypoints array
                swing_keypoints = {}
                # Dictionary to store Frame objects for later metrics storage
                frame_objects = {}
                # List to collect all processed frames
                processed_frames = []
                frame_index = 0
                for event_class, frame_idx in sorted(event_frames.items()):
                    # Calculate timestamp from frame index
                    # frame_idx is the frame index in the clipped video (0 to n_sequences * seq_len - 1)
                    # We need to map this back to the original video timestamp
                    timestamp = frame_idx / fps
                    event_label = EVENT_NAMES.get(event_class, f"Event_{event_class}")

                    logger.info(
                        f"Extracting frame for event {event_class} ({event_label}): "
                        f"frame_idx={frame_idx}, timestamp={timestamp:.3f}s, fps={fps}"
                    )

                    frame_path = os.path.join(temp_dir, f"frame_{frame_index}.jpg")

                    # Extract frame using ffmpeg
                    if not extract_single_frame(temp_video_path, timestamp, frame_path):
                        logger.error(
                            f"Failed to extract frame for event {event_class} "
                            f"({event_label}) at {timestamp}s"
                        )
                        continue

                    # Get frame dimensions and load image
                    with Image.open(frame_path) as img:
                        width, height = img.size
                        # Load image for pose estimation
                        image_array = np.array(img.convert("RGB"))

                    # Run pose estimation
                    try:
                        logger.info(f"Running pose estimation on frame {frame_index}")
                        keypoints = estimate_pose(
                            image_array,
                            model_name=settings.pose_model_name,
                            input_size=settings.pose_input_size,
                        )

                        # Draw pose overlay on image
                        annotated_image = draw_pose_overlay(
                            image_array,
                            keypoints,
                            keypoint_threshold=settings.pose_keypoint_threshold,
                        )

                        # Convert annotated image back to PIL Image for saving
                        annotated_pil = Image.fromarray(annotated_image)

                        # Save annotated frame to temporary file
                        annotated_frame_path = os.path.join(
                            temp_dir, f"frame_{frame_index}_annotated.jpg"
                        )
                        annotated_pil.save(annotated_frame_path, "JPEG", quality=95)

                        # Convert keypoints to JSON string for database storage
                        # keypoints is [1, 1, 17, 3] numpy array, convert to list
                        keypoints_list = keypoints.tolist()
                        keypoints_json = json.dumps(keypoints_list)

                        # Use annotated frame for upload
                        frame_to_upload = annotated_frame_path
                        logger.info(
                            f"Pose estimation completed for frame {frame_index}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to run pose estimation on frame {frame_index}: {e}",
                            exc_info=True,
                        )
                        # Fall back to original frame if pose estimation fails
                        frame_to_upload = frame_path
                        keypoints_json = None

                    # Save frame locally instead of uploading to S3
                    frame_filename = f"event_{event_class}_{event_label}.jpg"
                    local_frame_path = os.path.join(local_frames_dir, frame_filename)
                    
                    # Copy frame to local storage
                    shutil.copy2(frame_to_upload, local_frame_path)
                    logger.info(f"Saved frame locally: {local_frame_path}")
                    
                    # Store relative path in s3_key field (for tracking, actual S3 upload happens in background)
                    frame_s3_key = f"frames/{video_id}/{frame_filename}"
                    # Also track the actual S3 backup path
                    frame_s3_backup_key = f"videos/{video_id}/frames/{frame_filename}"

                    # Save frame metadata to database
                    frame = Frame(
                        video_id=video_id,
                        index=frame_index,
                        s3_key=frame_s3_key,  # Store relative local path
                        width=width,
                        height=height,
                        event_label=event_label,
                        event_class=event_class,
                        pose_keypoints=keypoints_json,
                    )
                    db.add(frame)
                    processed_frames.append(frame)
                    logger.info(
                        f"Saved frame {frame_index}: {event_label} "
                        f"(class {event_class}) at video frame {frame_idx} "
                        f"(timestamp {timestamp:.3f}s, fps {fps:.2f}) "
                        f"-> Local: {local_frame_path} "
                        f"({width}x{height})"
                    )

                    # Store keypoints for metrics computation (only for the 5 key positions)
                    # Event classes: 0=Address, 3=Top, 4=Mid-downswing, 5=Impact, 7=Finish
                    if event_class in [0, 3, 4, 5, 7]:  # address, top, mid_ds, impact, finish
                        position_map = {0: "address", 3: "top", 4: "mid_ds", 5: "impact", 7: "finish"}
                        position_name = position_map[event_class]
                        if keypoints_json is not None:
                            # Convert JSON back to numpy array for metrics computation
                            keypoints_array = np.array(json.loads(keypoints_json))
                            swing_keypoints[position_name] = keypoints_array
                            frame_objects[position_name] = frame

                    frame_index += 1

                # Compute swing metrics if we have all five key positions
                metrics_result = {}
                if len(swing_keypoints) == 5:
                    try:
                        logger.info("Computing swing metrics")
                        metrics_result = compute_metrics(swing_keypoints)

                        # Store metrics in each frame record
                        for position_name, frame_obj in frame_objects.items():
                            if position_name in metrics_result:
                                position_metrics = metrics_result[position_name]
                                # Convert numpy nan to None for JSON serialization
                                metrics_dict = {
                                    k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                    for k, v in position_metrics.items()
                                }
                                frame_obj.swing_metrics = json.dumps(metrics_dict)
                                logger.info(
                                    f"Stored metrics for {position_name}: {metrics_dict}"
                                )
                    except Exception as e:
                        logger.error(
                            f"Failed to compute swing metrics: {e}",
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        f"Not all key positions found for metrics computation. "
                        f"Found: {list(swing_keypoints.keys())}"
                    )

                db.commit()
                video.status = "processed"
                db.commit()
                logger.info(f"Successfully processed video: {video_id}")
                
                # Return frames and metrics data
                return {
                    "frames": processed_frames,
                    "metrics": metrics_result
                }

        finally:
            # Clean up temp video file only if we created it (downloaded from S3)
            # If temp_file_owned is False, the file is managed by the upload endpoint
            if temp_file_owned and temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                    logger.debug(f"Cleaned up temp video file: {temp_video_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp video file {temp_video_path}: {e}")
            elif temp_video_path and not temp_file_owned:
                # Clean up temp file from upload endpoint after processing
                try:
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
                        logger.debug(f"Cleaned up temp video file from upload: {temp_video_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp video file {temp_video_path}: {e}")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
        try:
            video.status = "failed"
            db.commit()
        except Exception:
            pass
        return None


def upload_video_and_frames_to_s3(video_id: UUID, video_path: str, frames: List[Frame]) -> None:
    """
    Upload video and frames to S3 for backup.
    This function is designed to be called from BackgroundTasks.
    
    Args:
        video_id: UUID of the video
        video_path: Path to the video file
        frames: List of Frame objects to upload
    """
    logger.info(f"Starting background S3 upload for video {video_id}")
    
    try:
        # Get video record to get s3_key
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.models.video import Video
        from app.core.config import settings
        
        engine = create_engine(settings.db_url)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            if not video:
                logger.error(f"Video not found for S3 upload: {video_id}")
                return
            
            # Upload video to S3
            if video_path and os.path.exists(video_path):
                try:
                    logger.info(f"Uploading video to S3: {video.s3_key}")
                    with open(video_path, "rb") as video_file:
                        # Determine content type from s3_key extension
                        content_type = "video/mp4"  # default
                        if video.s3_key.endswith(".mov") or video.s3_key.endswith(".MOV"):
                            content_type = "video/quicktime"
                        
                        success = s3_client.upload_file(
                            video_file,
                            video.s3_key,
                            content_type
                        )
                        if success:
                            logger.info(f"Successfully uploaded video to S3: {video.s3_key}")
                        else:
                            logger.warning(f"Failed to upload video to S3: {video.s3_key}")
                except Exception as e:
                    logger.error(f"Error uploading video to S3: {e}", exc_info=True)
            
            # Upload frames to S3
            for frame in frames:
                try:
                    # Extract filename from s3_key (which contains relative path: frames/{video_id}/filename.jpg)
                    frame_filename = os.path.basename(frame.s3_key)
                    frame_s3_key = f"videos/{video_id}/frames/{frame_filename}"
                    
                    # Get local frame path
                    local_frame_path = os.path.join(settings.frames_dir, str(video_id), frame_filename)
                    
                    if os.path.exists(local_frame_path):
                        with open(local_frame_path, "rb") as frame_file:
                            success = s3_client.upload_file(
                                frame_file,
                                frame_s3_key,
                                "image/jpeg"
                            )
                            if success:
                                logger.debug(f"Uploaded frame to S3: {frame_s3_key}")
                            else:
                                logger.warning(f"Failed to upload frame to S3: {frame_s3_key}")
                    else:
                        logger.warning(f"Local frame not found: {local_frame_path}")
                except Exception as e:
                    logger.error(f"Error uploading frame {frame.id} to S3: {e}", exc_info=True)
            
            logger.info(f"Completed background S3 upload for video {video_id}")
        finally:
            db.close()
        
        # Clean up temp video file after S3 upload
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logger.debug(f"Cleaned up temp video file after S3 upload: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp video file {video_path}: {e}")
    except Exception as e:
        logger.error(f"Error in background S3 upload for video {video_id}: {e}", exc_info=True)


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Failed to get video duration: {e}")
        return None


def get_video_fps(video_path: str) -> Optional[float]:
    """Get video frame rate (FPS) using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse fraction (e.g., "30/1" -> 30.0)
        fraction = result.stdout.strip()
        if "/" in fraction:
            num, den = map(float, fraction.split("/"))
            return num / den if den != 0 else None
        return float(fraction)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Failed to get video FPS: {e}")
        return None


def extract_single_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame at the given timestamp using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-ss",
                str(timestamp),
                "-vframes",
                "1",
                "-q:v",
                "2",  # High quality
                "-y",  # Overwrite output
                output_path,
            ],
            capture_output=True,
            check=True,
        )
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return False

