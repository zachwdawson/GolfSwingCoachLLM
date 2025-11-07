import logging
import subprocess
import tempfile
import os
from io import BytesIO
from typing import Optional
from uuid import UUID
from PIL import Image
from sqlalchemy.orm import Session
from app.models.video import Video
from app.models.frame import Frame
from app.services.aws import s3_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Number of frames to extract evenly spaced
NUM_FRAMES = 9


def extract_frames(video_id: UUID, db: Session) -> bool:
    """
    Extract N evenly spaced frames from a video using ffmpeg.
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

                # Get video duration using ffprobe
                duration = get_video_duration(temp_video_path)
                if duration is None:
                    logger.error(f"Failed to get video duration: {video_id}")
                    video.status = "failed"
                    db.commit()
                    return False

                # Extract frames evenly spaced
                frame_times = [duration * i / (NUM_FRAMES + 1) for i in range(1, NUM_FRAMES + 1)]
                logger.info(f"Extracting {NUM_FRAMES} frames at times: {frame_times}")

                # Create temp directory for frames
                with tempfile.TemporaryDirectory() as temp_dir:
                    for i, frame_time in enumerate(frame_times):
                        frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")
                        
                        # Extract frame using ffmpeg
                        if not extract_single_frame(temp_video_path, frame_time, frame_path):
                            logger.error(f"Failed to extract frame {i} at {frame_time}s")
                            continue

                        # Get frame dimensions
                        with Image.open(frame_path) as img:
                            width, height = img.size

                        # Upload frame to S3
                        frame_s3_key = f"videos/{video_id}/frames/frame_{i}.jpg"
                        with open(frame_path, "rb") as frame_file:
                            success = s3_client.upload_file(
                                frame_file,
                                frame_s3_key,
                                "image/jpeg",
                            )
                            if not success:
                                logger.error(f"Failed to upload frame {i} to S3")
                                continue

                        # Save frame metadata to database
                        frame = Frame(
                            video_id=video_id,
                            index=i,
                            s3_key=frame_s3_key,
                            width=width,
                            height=height,
                        )
                        db.add(frame)
                        logger.info(f"Saved frame {i}: {frame_s3_key} ({width}x{height})")

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

