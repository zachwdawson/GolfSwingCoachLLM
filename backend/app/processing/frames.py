import logging
import subprocess
import tempfile
import os
import time
import json
import numpy as np
from typing import Optional
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


def extract_frames(video_id: UUID, db: Session) -> bool:
    """
    Extract event frames from a video using model-based labeling.
    Saves frames to S3 and writes metadata to database.
    """
    try:
        # Get video record
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video not found: {video_id}")
            return False

        # Update video status
        video.status = "processing"
        db.commit()

        # Load model
        model = get_model()
        if model is None:
            logger.error("Model not available, cannot extract frames")
            video.status = "failed"
            db.commit()
            return False

        # Download video from S3 to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video_path = temp_video.name
            try:
                # Download from S3
                if not s3_client.download_file(video.s3_key, temp_video_path):
                    logger.error(f"Failed to download video from S3: {video.s3_key}")
                    video.status = "failed"
                    db.commit()
                    return False
                logger.info(f"Downloaded video from S3: {video.s3_key}")

                # Get video FPS for timestamp calculation
                fps = get_video_fps(temp_video_path)
                if fps is None:
                    logger.error(f"Failed to get video FPS: {video_id}")
                    video.status = "failed"
                    db.commit()
                    return False

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
                    return False

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

                        # Upload frame to S3
                        frame_s3_key = (
                            f"videos/{video_id}/frames/event_{event_class}_{event_label}.jpg"
                        )
                        with open(frame_to_upload, "rb") as frame_file:
                            success = s3_client.upload_file(
                                frame_file,
                                frame_s3_key,
                                "image/jpeg",
                            )
                            if not success:
                                logger.error(f"Failed to upload frame {frame_index} to S3")
                                continue

                        # Save frame metadata to database
                        frame = Frame(
                            video_id=video_id,
                            index=frame_index,
                            s3_key=frame_s3_key,
                            width=width,
                            height=height,
                            event_label=event_label,
                            event_class=event_class,
                            pose_keypoints=keypoints_json,
                        )
                        db.add(frame)
                        logger.info(
                            f"Saved frame {frame_index}: {event_label} "
                            f"(class {event_class}) at video frame {frame_idx} "
                            f"(timestamp {timestamp:.3f}s, fps {fps:.2f}) "
                            f"-> S3 key: {frame_s3_key} "
                            f"({width}x{height})"
                        )

                        # Store keypoints for metrics computation (only for the 4 key positions)
                        if event_class in [0, 3, 5, 7]:  # address, top, impact, finish
                            position_map = {0: "address", 3: "top", 5: "impact", 7: "finish"}
                            position_name = position_map[event_class]
                            if keypoints_json is not None:
                                # Convert JSON back to numpy array for metrics computation
                                keypoints_array = np.array(json.loads(keypoints_json))
                                swing_keypoints[position_name] = keypoints_array
                                frame_objects[position_name] = frame

                        frame_index += 1

                    # Compute swing metrics if we have all four key positions
                    if len(swing_keypoints) == 4:
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
                    return True

            finally:
                # Clean up temp video file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
        try:
            video.status = "failed"
            db.commit()
        except Exception:
            pass
        return False


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

