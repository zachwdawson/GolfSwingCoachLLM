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
    
    # Create test frames with event_class values (Address, Top, Mid-downswing, Impact, Finish)
    event_classes = [0, 3, 4, 5, 7]
    event_labels = ["Address", "Top", "Mid-downswing (arm parallel)", "Impact", "Finish"]
    for i, (event_class, event_label) in enumerate(zip(event_classes[:3], event_labels[:3])):
        frame = Frame(
            video_id=video_id,
            index=i,
            s3_key=f"videos/{video_id}/frames/event_{event_class}_{event_label}.jpg",
            width=1920,
            height=1080,
            event_class=event_class,
            event_label=event_label,
        )
        db.add(frame)
    
    db.commit()
    db.close()

    response = client.get(f"/videos/{video_id}/frames")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == str(video_id)
    assert len(data["frames"]) == 3  # Should return 3 frames (Address, Top, Mid-downswing)
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
        # Mock model service to return a mock model
        with patch("app.processing.frames.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.device.type = "cpu"
            mock_get_model.return_value = mock_model
            
            # Mock video preprocessing and inference
            with patch("app.processing.frames.preprocess_video") as mock_preprocess:
                # Create a mock preprocessed video tensor
                import torch
                mock_video_tensor = torch.zeros(100, 112, 112, 3, dtype=torch.uint8)
                mock_preprocess.return_value = mock_video_tensor
                
                with patch("app.processing.frames.prepare_video_for_inference") as mock_prepare:
                    mock_video_tensor_prepared = torch.zeros(1, 3, 64, 112, 112, dtype=torch.float32)
                    mock_prepare.return_value = (mock_video_tensor_prepared, 1)
                    
                    with patch("app.processing.frames.process_video_for_events") as mock_process:
                        # Return mock event frames
                        mock_process.return_value = {i: i * 10 for i in range(8)}
                        
                        # Mock S3 client in frames module
                        with patch("app.processing.frames.s3_client") as mock_s3:
                            mock_s3.download_file.return_value = True
                            mock_s3.upload_file.return_value = True
                            
                            # Mock get_video_fps
                            with patch("app.processing.frames.get_video_fps") as mock_fps:
                                mock_fps.return_value = 30.0
                                
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
                                                # Mock pose estimation functions
                                                with patch("app.processing.frames.estimate_pose") as mock_estimate_pose:
                                                    import numpy as np
                                                    # Mock keypoints: [1, 1, 17, 3] format
                                                    mock_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
                                                    mock_keypoints[0, 0, :, 2] = 0.5  # Set confidence scores
                                                    mock_estimate_pose.return_value = mock_keypoints
                                                    
                                                    with patch("app.processing.frames.draw_pose_overlay") as mock_draw_pose:
                                                        # Mock annotated image (same size as input)
                                                        mock_annotated = np.zeros((1080, 1920, 3), dtype=np.uint8)
                                                        mock_draw_pose.return_value = mock_annotated
                                                        
                                                        # Run extraction
                                                        result = extract_frames(video_id, db)
                                                        
                                                        # Verify it completed
                                                        assert result is True
                                                        
                                                        # Verify video status updated
                                                        db.refresh(video)
                                                        assert video.status == "processed"
                                                        
                                                        # Verify frames were created
                                                        frames = db.query(Frame).filter(Frame.video_id == video_id).all()
                                                        assert len(frames) == 8  # Should have 8 event frames
                                                        
                                                        # Verify pose estimation was called for each frame
                                                        assert mock_estimate_pose.call_count == 8
                                                        assert mock_draw_pose.call_count == 8
                                                        
                                                        # Verify pose_keypoints are stored in database
                                                        for frame in frames:
                                                            assert frame.pose_keypoints is not None
                                                            import json
                                                            keypoints_data = json.loads(frame.pose_keypoints)
                                                            assert len(keypoints_data) == 1  # [1, 1, 17, 3]
                                                            assert len(keypoints_data[0]) == 1
                                                            assert len(keypoints_data[0][0]) == 17  # 17 keypoints
    finally:
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)


def test_frame_extraction_with_pose_estimation_failure(client, mock_ffmpeg, mock_pil, test_db):
    """Test frame extraction falls back to original frame when pose estimation fails."""
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
        # Mock model service to return a mock model
        with patch("app.processing.frames.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.device.type = "cpu"
            mock_get_model.return_value = mock_model
            
            # Mock video preprocessing and inference
            with patch("app.processing.frames.preprocess_video") as mock_preprocess:
                import torch
                mock_video_tensor = torch.zeros(100, 112, 112, 3, dtype=torch.uint8)
                mock_preprocess.return_value = mock_video_tensor
                
                with patch("app.processing.frames.prepare_video_for_inference") as mock_prepare:
                    mock_video_tensor_prepared = torch.zeros(1, 3, 64, 112, 112, dtype=torch.float32)
                    mock_prepare.return_value = (mock_video_tensor_prepared, 1)
                    
                    with patch("app.processing.frames.process_video_for_events") as mock_process:
                        mock_process.return_value = {i: i * 10 for i in range(8)}
                        
                        with patch("app.processing.frames.s3_client") as mock_s3:
                            mock_s3.download_file.return_value = True
                            mock_s3.upload_file.return_value = True
                            
                            with patch("app.processing.frames.get_video_fps") as mock_fps:
                                mock_fps.return_value = 30.0
                                
                                with patch("app.processing.frames.tempfile") as mock_tempfile:
                                    def mock_named_tempfile(*args, **kwargs):
                                        mock_temp = MagicMock()
                                        mock_temp.name = temp_video_path
                                        mock_temp.__enter__ = lambda self: self
                                        mock_temp.__exit__ = lambda *args: None
                                        return mock_temp
                                    
                                    mock_tempfile.NamedTemporaryFile.side_effect = mock_named_tempfile
                                    
                                    mock_temp_dir = MagicMock()
                                    mock_temp_dir.__enter__.return_value = "/tmp/frames"
                                    mock_temp_dir.__exit__.return_value = None
                                    mock_tempfile.TemporaryDirectory.return_value = mock_temp_dir
                                    
                                    with patch("app.processing.frames.os.path.exists") as mock_exists:
                                        def exists_side_effect(path):
                                            if path == temp_video_path:
                                                return True
                                            return path.endswith(".jpg")
                                        mock_exists.side_effect = exists_side_effect
                                        
                                        with patch("builtins.open", create=True) as mock_open:
                                            def open_side_effect(path, mode="r"):
                                                if "frame" in path and mode == "rb":
                                                    return io.BytesIO(b"fake image data")
                                                return open(path, mode) if os.path.exists(path) else MagicMock()
                                            mock_open.side_effect = open_side_effect
                                            
                                            with patch("app.processing.frames.os.unlink"):
                                                # Mock pose estimation to raise an exception
                                                with patch("app.processing.frames.estimate_pose") as mock_estimate_pose:
                                                    mock_estimate_pose.side_effect = Exception("Pose estimation failed")
                                                    
                                                    # Run extraction
                                                    result = extract_frames(video_id, db)
                                                    
                                                    # Verify it completed (should fall back to original frames)
                                                    assert result is True
                                                    
                                                    # Verify video status updated
                                                    db.refresh(video)
                                                    assert video.status == "processed"
                                                    
                                                    # Verify frames were created
                                                    frames = db.query(Frame).filter(Frame.video_id == video_id).all()
                                                    assert len(frames) == 8
                                                    
                                                    # Verify pose_keypoints are None when pose estimation fails
                                                    for frame in frames:
                                                        assert frame.pose_keypoints is None
    finally:
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

