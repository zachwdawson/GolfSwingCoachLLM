"""
Tests for pose estimation module.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from app.processing import pose_estimation
from app.processing.pose_estimation import (
    load_movenet_model,
    estimate_pose,
    draw_pose_overlay,
    SKELETON,
)


@pytest.fixture(autouse=True)
def reset_pose_cache():
    """Reset the pose estimation model cache before each test."""
    pose_estimation._movenet_model = None
    pose_estimation._movenet_input_size = None
    yield
    # Clean up after test
    pose_estimation._movenet_model = None
    pose_estimation._movenet_input_size = None


def test_skeleton_connections():
    """Test that SKELETON has correct number of connections."""
    # MoveNet has 17 keypoints with 18 connections
    assert len(SKELETON) == 18
    # All connections should be tuples of two integers
    for connection in SKELETON:
        assert len(connection) == 2
        assert all(isinstance(i, int) for i in connection)
        # Keypoint indices should be in range [0, 16] for 17 keypoints
        assert 0 <= connection[0] <= 16
        assert 0 <= connection[1] <= 16


@patch("app.processing.pose_estimation.hub.load")
def test_load_movenet_model_thunder(mock_hub_load):
    """Test loading MoveNet Thunder model."""
    # Mock TensorFlow Hub module
    mock_module = MagicMock()
    mock_signature = MagicMock()
    mock_output = MagicMock()
    mock_output.numpy.return_value = np.zeros((1, 1, 17, 3), dtype=np.float32)
    mock_signature.return_value = {"output_0": mock_output}
    mock_module.signatures = {"serving_default": mock_signature}
    mock_hub_load.return_value = mock_module
    
    movenet_fn, input_size = load_movenet_model("movenet_thunder")
    
    assert input_size == 256
    assert callable(movenet_fn)
    
    # Test that function can be called
    test_image = np.zeros((1, 256, 256, 3), dtype=np.int32)
    result = movenet_fn(test_image)
    assert result.shape == (1, 1, 17, 3)


@patch("app.processing.pose_estimation.hub.load")
def test_load_movenet_model_lightning(mock_hub_load):
    """Test loading MoveNet Lightning model."""
    mock_module = MagicMock()
    mock_signature = MagicMock()
    mock_output = MagicMock()
    mock_output.numpy.return_value = np.zeros((1, 1, 17, 3), dtype=np.float32)
    mock_signature.return_value = {"output_0": mock_output}
    mock_module.signatures = {"serving_default": mock_signature}
    mock_hub_load.return_value = mock_module
    
    movenet_fn, input_size = load_movenet_model("movenet_lightning")
    
    assert input_size == 192
    assert callable(movenet_fn)


@patch("app.processing.pose_estimation.load_movenet_model")
def test_estimate_pose_numpy_array(mock_load_model):
    """Test pose estimation with numpy array input."""
    # Mock model function
    def mock_movenet(input_image):
        return np.zeros((1, 1, 17, 3), dtype=np.float32)
    
    mock_load_model.return_value = (mock_movenet, 256)
    
    # Create test image
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    keypoints = estimate_pose(test_image, model_name="movenet_thunder")
    
    assert keypoints.shape == (1, 1, 17, 3)
    mock_load_model.assert_called_once_with("movenet_thunder")


@patch("app.processing.pose_estimation.load_movenet_model")
def test_estimate_pose_pil_image(mock_load_model):
    """Test pose estimation with PIL Image input."""
    def mock_movenet(input_image):
        return np.zeros((1, 1, 17, 3), dtype=np.float32)
    
    mock_load_model.return_value = (mock_movenet, 256)
    
    # Create test PIL image
    test_image = Image.new("RGB", (1920, 1080), color="red")
    
    keypoints = estimate_pose(test_image, model_name="movenet_thunder")
    
    assert keypoints.shape == (1, 1, 17, 3)


def test_draw_pose_overlay_numpy_array():
    """Test drawing pose overlay on numpy array."""
    # Create test image
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Create mock keypoints with some visible keypoints
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Set some keypoints to be visible (confidence > 0.11)
    keypoints[0, 0, 0, :] = [0.5, 0.5, 0.5]  # y, x, score
    keypoints[0, 0, 1, :] = [0.5, 0.6, 0.5]
    keypoints[0, 0, 2, :] = [0.5, 0.4, 0.5]
    
    annotated = draw_pose_overlay(test_image, keypoints, keypoint_threshold=0.11)
    
    assert annotated.shape == test_image.shape
    assert annotated.dtype == np.uint8
    # Image should be modified (not identical)
    assert not np.array_equal(annotated, test_image)


def test_draw_pose_overlay_pil_image():
    """Test drawing pose overlay on PIL Image."""
    test_image = Image.new("RGB", (1920, 1080), color="blue")
    
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    keypoints[0, 0, 0, :] = [0.5, 0.5, 0.5]
    
    annotated = draw_pose_overlay(test_image, keypoints)
    
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (1080, 1920, 3)
    assert annotated.dtype == np.uint8


def test_draw_pose_overlay_low_confidence():
    """Test that low confidence keypoints are not drawn."""
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Create keypoints with low confidence
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    keypoints[0, 0, 0, :] = [0.5, 0.5, 0.05]  # Confidence below threshold
    
    annotated = draw_pose_overlay(test_image, keypoints, keypoint_threshold=0.11)
    
    # Image should remain unchanged since no keypoints are drawn
    assert np.array_equal(annotated, test_image)


def test_draw_pose_overlay_skeleton_connections():
    """Test that skeleton connections are drawn between visible keypoints."""
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Set two connected keypoints to be visible
    # Keypoint 0 and 1 are connected according to SKELETON
    keypoints[0, 0, 0, :] = [0.5, 0.5, 0.5]  # y, x, score
    keypoints[0, 0, 1, :] = [0.5, 0.6, 0.5]
    
    annotated = draw_pose_overlay(test_image, keypoints, keypoint_threshold=0.11)
    
    # Image should be modified
    assert not np.array_equal(annotated, test_image)

