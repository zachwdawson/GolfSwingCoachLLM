import logging
import threading
from queue import Queue
from uuid import UUID
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.processing.frames import extract_frames
from app.core.config import settings
from app.models.video import Base
from app.models.frame import Frame  # Import to ensure table is created

logger = logging.getLogger(__name__)

# Simple in-process task queue
task_queue = Queue()
worker_thread = None
worker_running = False


def start_worker():
    """Start the background worker thread."""
    global worker_thread, worker_running
    if worker_running:
        return

    worker_running = True
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    logger.info("Frame processing worker started")


def get_session_local():
    """Get database session factory."""
    engine = create_engine(settings.db_url)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def worker_loop():
    """Worker loop that processes tasks from the queue."""
    SessionLocal = get_session_local()
    while worker_running:
        try:
            video_id = task_queue.get(timeout=1)
            if video_id is None:
                continue

            logger.info(f"Processing video: {video_id}")
            db = SessionLocal()
            try:
                extract_frames(video_id, db)
            finally:
                db.close()
                task_queue.task_done()
        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)


def enqueue_frame_extraction(video_id: UUID):
    """Enqueue a video for frame extraction."""
    task_queue.put(video_id)
    logger.info(f"Enqueued video for processing: {video_id}")

