"""Model service for lazy loading and caching the frame classifier model."""
import logging
import threading
from typing import Optional, TYPE_CHECKING

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from app.ml.model import GolfDBFrameClassifier

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global model instance and lock for thread-safe access
_model: Optional["GolfDBFrameClassifier"] = None
_model_lock = threading.Lock()


def get_model() -> Optional["GolfDBFrameClassifier"]:
    """
    Get or load the model singleton.

    Returns:
        Loaded model instance, or None if model path is not configured or torch is not available
    """
    global _model

    if not TORCH_AVAILABLE:
        logger.warning("torch is not installed. Model functionality is unavailable.")
        return None

    if _model is not None:
        return _model

    if not settings.model_checkpoint_path:
        logger.warning("Model checkpoint path not configured")
        return None

    with _model_lock:
        # Double-check pattern
        if _model is not None:
            return _model

        try:
            from app.ml.model import GolfDBFrameClassifier, load_model

            # Determine device
            device = settings.model_device
            if not device or device == "":
                # Auto-detect: use CUDA if available, otherwise CPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Auto-detected device: {device}")
            elif device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"
            # else: use the specified device

            _model = load_model(settings.model_checkpoint_path, device)
            return _model
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return None


def clear_model():
    """Clear the cached model (useful for testing or reloading)."""
    global _model
    with _model_lock:
        _model = None

