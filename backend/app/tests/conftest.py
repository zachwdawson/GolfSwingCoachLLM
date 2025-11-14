"""Pytest configuration and fixtures."""
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest

# Set TESTING environment variable to prevent worker thread from starting
os.environ["TESTING"] = "1"


@pytest.fixture(autouse=True)
def patch_settings():
    """Automatically patch settings for all tests to use temp directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        frames_dir = os.path.join(temp_dir, "frames")
        model_dir = os.path.join(temp_dir, "ml")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a dummy model file to avoid FileNotFoundError
        dummy_model_path = os.path.join(model_dir, "latest_model__epoch_20.pth")
        with open(dummy_model_path, "wb") as f:
            f.write(b"dummy model data")
        
        # Create a default mock model for tests that don't explicitly mock get_model
        mock_model = MagicMock()
        mock_model.device.type = "cpu"
        # Mock parameters() to return an iterable with a device attribute
        mock_params = MagicMock()
        mock_params.device.type = "cpu"
        mock_model.parameters.return_value = [mock_params]
        
        with patch("app.core.config.settings.frames_dir", frames_dir), \
             patch("app.core.config.settings.model_checkpoint_path", dummy_model_path), \
             patch("app.ml.service.get_model", return_value=mock_model), \
             patch("app.processing.frames.get_model", return_value=mock_model):
            yield

