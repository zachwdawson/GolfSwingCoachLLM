import io
import uuid
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.api.upload import get_db
from app.main import app
from app.models.video import Base, Video
from app.models.frame import Frame

TEST_DB_URL = "sqlite:///./test_frames.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(test_db):
    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_s3_client():
    with patch("app.api.upload.s3_client") as mock:
        mock.upload_file.return_value = True
        mock.download_file.return_value = True
        mock.generate_presigned_url.return_value = "https://example.com/presigned-url"
        yield mock


@pytest.fixture
def mock_ffmpeg():
    """Mock ffmpeg subprocess calls."""
    with patch("app.processing.frames.subprocess.run") as mock_run:
        # Mock ffprobe for duration
        def mock_run_side_effect(*args, **kwargs):
            if "ffprobe" in args[0]:
                mock_result = MagicMock()
                mock_result.stdout = "10.5\n"  # 10.5 seconds duration
                mock_result.returncode = 0
                return mock_result
            elif "ffmpeg" in args[0]:
                mock_result = MagicMock()
                mock_result.returncode = 0
                return mock_result
            return MagicMock()

        mock_run.side_effect = mock_run_side_effect
        yield mock_run


@pytest.fixture
def mock_pil():
    """Mock PIL Image for frame dimensions."""
    with patch("app.processing.frames.Image") as mock_image:
        mock_img = MagicMock()
        mock_img.size = (1920, 1080)  # 1920x1080
        mock_image.open.return_value.__enter__.return_value = mock_img
        yield mock_image


def test_get_frames_not_found(client):
    """Test getting frames for non-existent video returns 404."""
    video_id = uuid.uuid4()
    response = client.get(f"/videos/{video_id}/frames")
    assert response.status_code == 404


def test_get_frames_empty(client, mock_s3_client, test_db):
    """Test getting frames when none exist."""
    video_id = uuid.uuid4()
    s3_key = f"videos/{video_id}/raw.mp4"
    video = Video(id=video_id, s3_key=s3_key, status="uploaded")

    db = TestSessionLocal()
    db.add(video)
    db.commit()
    db.close()

    response = client.get(f"/videos/{video_id}/frames")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == str(video_id)
    assert data["frames"] == []


def test_get_frames_success(client, mock_s3_client, test_db):
    """Test getting frames with presigned URLs."""
    video_id = uuid.uuid4()
    s3_key = f"videos/{video_id}/raw.mp4"
    video = Video(id=video_id, s3_key=s3_key, status="processed")

    db = TestSessionLocal()
    db.add(video)
    
    # Create test frames
    for i in range(3):
        frame = Frame(
            video_id=video_id,
            index=i,
            s3_key=f"videos/{video_id}/frames/frame_{i}.jpg",
            width=1920,
            height=1080,
        )
        db.add(frame)
    
    db.commit()
    db.close()

    response = client.get(f"/videos/{video_id}/frames")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == str(video_id)
    assert len(data["frames"]) == 3
    assert all("url" in frame for frame in data["frames"])
    assert all(frame["width"] == 1920 for frame in data["frames"])
    assert all(frame["height"] == 1080 for frame in data["frames"])


def test_frame_extraction_mocked(client, mock_ffmpeg, mock_pil, test_db):
    """Test frame extraction with mocked ffmpeg and S3."""
    from app.processing.frames import extract_frames
    
    video_id = uuid.uuid4()
    s3_key = f"videos/{video_id}/raw.mp4"
    video = Video(id=video_id, s3_key=s3_key, status="uploaded")

    db = TestSessionLocal()
    db.add(video)
    db.commit()
    
    # Mock S3 download to create a temp file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video_path = temp_video.name
        temp_video.write(b"fake video content")
    
    try:
        # Mock S3 client in frames module
        with patch("app.processing.frames.s3_client") as mock_s3:
            mock_s3.download_file.return_value = True
            mock_s3.upload_file.return_value = True
            
            # Mock tempfile to use our actual temp file
            with patch("app.processing.frames.tempfile") as mock_tempfile:
                # Mock NamedTemporaryFile to return our temp file
                def mock_named_tempfile(*args, **kwargs):
                    mock_temp = MagicMock()
                    mock_temp.name = temp_video_path
                    mock_temp.__enter__ = lambda self: self
                    mock_temp.__exit__ = lambda *args: None
                    return mock_temp
                
                mock_tempfile.NamedTemporaryFile.side_effect = mock_named_tempfile
                
                # Mock TemporaryDirectory
                mock_temp_dir = MagicMock()
                mock_temp_dir.__enter__.return_value = "/tmp/frames"
                mock_temp_dir.__exit__.return_value = None
                mock_tempfile.TemporaryDirectory.return_value = mock_temp_dir
                
                # Mock os.path.exists for frame files
                with patch("app.processing.frames.os.path.exists") as mock_exists:
                    def exists_side_effect(path):
                        if path == temp_video_path:
                            return True
                        return path.endswith(".jpg")
                    mock_exists.side_effect = exists_side_effect
                    
                    # Mock open for reading frame files
                    with patch("builtins.open", create=True) as mock_open:
                        def open_side_effect(path, mode="r"):
                            if "frame" in path and mode == "rb":
                                return io.BytesIO(b"fake image data")
                            return open(path, mode) if os.path.exists(path) else MagicMock()
                        mock_open.side_effect = open_side_effect
                        
                        # Mock os.unlink to prevent errors
                        with patch("app.processing.frames.os.unlink"):
                            # Run extraction
                            result = extract_frames(video_id, db)
                            
                            # Verify it completed
                            assert result is True
                            
                            # Verify video status updated
                            db.refresh(video)
                            assert video.status == "processed"
                            
                            # Verify frames were created
                            frames = db.query(Frame).filter(Frame.video_id == video_id).all()
                            assert len(frames) > 0
    finally:
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

