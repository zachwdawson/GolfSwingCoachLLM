"""Test model inference with actual video file."""
import os
import uuid
from app.core.config import settings
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Try to import torch, skip tests if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("torch is not installed. PyTorch requires Python 3.11 or 3.12. "
                "Python 3.14 is not yet supported. Please install Python 3.12 or 3.11 "
                "and recreate the virtual environment.", allow_module_level=True)

from app.ml.model import GolfDBFrameClassifier, load_model
from app.ml.preprocessing import preprocess_video, prepare_video_for_inference
from app.ml.inference import process_video_for_events, EVENT_NAMES
from app.ml.service import get_model, clear_model


# Path to test video file
TEST_VIDEO_PATH = Path(__file__).parent / "12.mp4"


def create_random_model(num_classes: int = 9, device: str = "cpu") -> GolfDBFrameClassifier:
    """Create a randomly initialized model for testing."""
    model = load_model("app/ml/latest_model__epoch_20.pth", device)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def random_model():
    """Fixture to provide a randomly initialized model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_random_model(num_classes=9, device=device)
    yield model
    # Cleanup if needed
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def test_video_path():
    """Fixture to provide the test video path."""
    if not TEST_VIDEO_PATH.exists():
        pytest.skip(f"Test video file not found: {TEST_VIDEO_PATH}")
    return str(TEST_VIDEO_PATH)


def test_preprocess_video(test_video_path):
    """Test video preprocessing."""
    preprocessed = preprocess_video(test_video_path)
    
    # Verify shape: [T, H, W, C]
    assert len(preprocessed.shape) == 4
    assert preprocessed.shape[1] == 112  # Height
    assert preprocessed.shape[2] == 112  # Width
    assert preprocessed.shape[3] == 3    # RGB channels
    assert preprocessed.dtype == torch.uint8
    
    # Verify we have frames
    assert preprocessed.shape[0] > 0
    print(f"Preprocessed video shape: {preprocessed.shape}")


def test_prepare_video_for_inference(test_video_path, random_model):
    """Test video preparation for inference."""
    preprocessed = preprocess_video(test_video_path)
    device = str(next(random_model.parameters()).device)
    
    seq_len = 64
    video_tensor, n_sequences = prepare_video_for_inference(
        preprocessed, seq_len, device
    )
    
    # Verify shape: [N_SEQUENCES, C, SEQ_LEN, H, W]
    assert len(video_tensor.shape) == 5
    assert video_tensor.shape[0] == n_sequences
    assert video_tensor.shape[1] == 3  # RGB channels
    assert video_tensor.shape[2] == seq_len
    assert video_tensor.shape[3] == 112  # Height
    assert video_tensor.shape[4] == 112  # Width
    assert video_tensor.dtype == torch.float32
    assert video_tensor.min() >= 0.0
    assert video_tensor.max() <= 1.0
    
    print(f"Prepared video tensor shape: {video_tensor.shape}, sequences: {n_sequences}")


def test_model_inference_on_video(test_video_path, random_model):
    """Test model inference on actual video file."""
    # Preprocess video
    preprocessed = preprocess_video(test_video_path)
    
    # Prepare for inference
    device = str(next(random_model.parameters()).device)
    seq_len = 64
    video_tensor, n_sequences = prepare_video_for_inference(
        preprocessed, seq_len, device
    )
    
    # Run inference
    event_frames = process_video_for_events(
        random_model, video_tensor, n_sequences, seq_len
    )
    
    # Verify we got event frames
    assert len(event_frames) == 8  # Should have 8 event classes
    
    # Verify all event classes are present
    for event_class in range(8):
        assert event_class in event_frames
        assert isinstance(event_frames[event_class], int)
        assert event_frames[event_class] >= 0
        assert event_frames[event_class] < n_sequences * seq_len
 
    # Ensure key events map to distinct frames (Address, Top, Impact, Finish)
    key_event_classes = [0, 3, 5, 7]
    key_event_frames = [event_frames[c] for c in key_event_classes]
    # assert len(set(key_event_frames)) == len(key_event_frames), (
    #     f"Key events should map to distinct frames. Got indices: {key_event_frames}"
    # )

    # Print results
    print("\nEvent frames identified:")
    for event_class, frame_idx in sorted(event_frames.items()):
        event_name = EVENT_NAMES.get(event_class, f"Event_{event_class}")
        print(f"  {event_name} (class {event_class}): frame {frame_idx}")


def test_full_pipeline_with_mocked_s3(test_video_path, random_model):
    """Test full pipeline with mocked S3 and database."""
    from app.processing.frames import extract_frames, get_video_fps
    from app.models.video import Video
    from app.models.frame import Frame
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.video import Base
    import tempfile
    import shutil
    
    # Create test database
    TEST_DB_URL = "sqlite:///./test_model_inference.db"
    test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    try:
        # Create video record
        db = TestSessionLocal()
        video_id = uuid.uuid4()
        video = Video(
            id=video_id,
            s3_key="test/video.mp4",
            status="uploaded"
        )
        db.add(video)
        db.commit()
        db.close()
        
        # Mock S3 client
        with patch("app.processing.frames.s3_client") as mock_s3:
            mock_s3.download_file.return_value = True
            mock_s3.upload_file.return_value = True
            
            # Copy test video to temp location for S3 download simulation
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video_path = temp_video.name
                shutil.copy(test_video_path, temp_video_path)
            
            try:
                # Mock model service to return our random model
                with patch("app.processing.frames.get_model", return_value=random_model):
                    # Mock get_video_fps to avoid ffprobe dependency
                    with patch("app.processing.frames.get_video_fps") as mock_fps:
                        mock_fps.return_value = 30.0
                        
                        # Mock tempfile to use our actual video file
                        with patch("app.processing.frames.tempfile") as mock_tempfile:
                            # Mock NamedTemporaryFile
                            def mock_named_tempfile(*args, **kwargs):
                                mock_temp = MagicMock()
                                mock_temp.name = temp_video_path
                                mock_temp.__enter__ = lambda self: self
                                mock_temp.__exit__ = lambda *args: None
                                return mock_temp
                            
                            mock_tempfile.NamedTemporaryFile.side_effect = mock_named_tempfile
                            
                            # Mock TemporaryDirectory
                            with tempfile.TemporaryDirectory() as temp_dir:
                                mock_temp_dir = MagicMock()
                                mock_temp_dir.__enter__.return_value = temp_dir
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
                                    import io
                                    with patch("builtins.open", create=True) as mock_open:
                                        def open_side_effect(path, mode="r"):
                                            if "frame" in path and mode == "rb":
                                                # Create a minimal JPEG header
                                                return io.BytesIO(
                                                    b"\xff\xd8\xff\xe0\x00\x10JFIF"
                                                    + b"\x00" * 1000  # Fake image data
                                                )
                                            return open(path, mode) if os.path.exists(path) else MagicMock()
                                        mock_open.side_effect = open_side_effect
                                        
                                        # Mock PIL Image
                                        from PIL import Image as PILImage
                                        with patch("app.processing.frames.Image") as mock_image:
                                            mock_img = MagicMock()
                                            mock_img.size = (1920, 1080)
                                            mock_image.open.return_value.__enter__.return_value = mock_img
                                            
                                            # Mock os.unlink
                                            with patch("app.processing.frames.os.unlink"):
                                                # Mock extract_single_frame to avoid ffmpeg dependency
                                                with patch("app.processing.frames.extract_single_frame") as mock_extract:
                                                    mock_extract.return_value = True
                                                    
                                                    # Run extraction
                                                    db = TestSessionLocal()
                                                    result = extract_frames(video_id, db)
                                                
                                                # Verify it completed
                                                assert result is True
                                                
                                                # Verify video status updated
                                                video = db.query(Video).filter(Video.id == video_id).first()
                                                assert video is not None
                                                assert video.status == "processed"
                                                
                                                # Verify frames were created
                                                frames = db.query(Frame).filter(Frame.video_id == video_id).all()
                                                assert len(frames) == 8  # Should have 8 event frames
                                                
                                                # Verify frame metadata
                                                for frame in frames:
                                                    assert frame.event_class is not None
                                                    assert frame.event_label is not None
                                                    assert frame.event_class in range(8)
                                                    assert frame.event_label in EVENT_NAMES.values()

                                                # Ensure distinct frames for key events (Address, Top, Impact, Finish)
                                                key_event_classes = {0, 3, 5, 7}
                                                key_frames = [
                                                    f for f in frames if f.event_class in key_event_classes
                                                ]
                                                # Use the database 'index' to verify distinct selections
                                                key_indices = [f.index for f in key_frames]
                                                assert len(key_frames) == 4
                                                assert len(set(key_indices)) == 4, (
                                                    f"Key event frames should be distinct. Got indices: {key_indices}"
                                                )
 
                                                print(f"\nSuccessfully extracted {len(frames)} event frames:")
                                                for frame in frames:
                                                    print(
                                                        f"  Frame {frame.index}: {frame.event_label} "
                                                        f"(class {frame.event_class})"
                                                    )
                                                
                                                db.close()
            finally:
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
    finally:
        # Clean up database
        Base.metadata.drop_all(bind=test_engine)
        if os.path.exists("test_model_inference.db"):
            os.unlink("test_model_inference.db")

