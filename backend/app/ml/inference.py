"""Model inference utilities for frame labeling."""
import logging
from typing import Optional

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from app.ml.model import GolfDBFrameClassifier
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    np = None  # type: ignore
    torch = None  # type: ignore
    F = None  # type: ignore
    GolfDBFrameClassifier = None  # type: ignore

logger = logging.getLogger(__name__)

# Event class names mapping
EVENT_NAMES = {
    0: "Address",
    1: "Toe-up",
    2: "Mid-backswing (arm parallel)",
    3: "Top",
    4: "Mid-downswing (arm parallel)",
    5: "Impact",
    6: "Mid-follow-through (shaft parallel)",
    7: "Finish",
}

NUM_EVENTS = 8


def predict_frame_labels(
    model: GolfDBFrameClassifier,
    video_tensor: torch.Tensor,
    n_sequences: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Run model inference on video tensor.

    Args:
        model: Loaded GolfDBFrameClassifier model
        video_tensor: Preprocessed video tensor [N_SEQUENCES, C, SEQ_LEN, H, W]
        n_sequences: Number of sequences
        seq_len: Sequence length

    Returns:
        Logits tensor of shape [N_SEQUENCES * SEQ_LEN, N_CLASSES]
    """
    with torch.no_grad():
        logits = model(video_tensor)  # [N_SEQUENCES, SEQ_LEN, N_CLASSES]
        logits = logits.reshape([n_sequences * seq_len, -1])  # [N_FRAMES, N_CLASSES]

    return logits


def extract_event_frames(
    logits: torch.Tensor,
    n_sequences: int,
    seq_len: int,
) -> dict[int, int]:
    """
    Extract frame indices for each event class from model predictions.
    
    Uses the original approach from AG_metrics.py:
    - Apply softmax to get probabilities
    - For each class, find the frame with maximum probability (argmax along dim=0)
    - Exclude 'no-event' class (last class)

    Args:
        logits: Model logits [N_FRAMES, N_CLASSES]
        n_sequences: Number of sequences processed
        seq_len: Sequence length

    Returns:
        Dictionary mapping event_class (0-7) to frame_index
    """
    # Apply softmax to the model prediction at each point in time
    # probabilities shape: [N_FRAMES, N_CLASSES]

    logits = logits.reshape([n_sequences * seq_len, -1])  # Predictions.

    # Apply softmax to the model prediction at each point in time.
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    event_estimates = torch.argmax(probabilities, dim=0).cpu()
    event_estimates = event_estimates[:-1]


    # probabilities = F.softmax(logits, dim=1)
    
    # # For each class (column), find which frame (row) has the maximum probability
    # # event_estimates shape: [N_CLASSES] where each element is the frame index with max prob for that class
    # event_estimates = torch.argmax(probabilities, dim=0).cpu()
    
    # # Exclude 'no-event' class (last class, index 8)
    # # We only want the 8 event classes (0-7)
    # event_estimates = event_estimates[:-1]  # Shape: [NUM_EVENTS] = [8]
    
    # Convert to dictionary mapping event_class (0-7) to frame_index
    event_frames = {}
    for event_class in range(NUM_EVENTS):
        frame_idx = event_estimates[event_class].item()
        event_frames[event_class] = frame_idx
        
        logger.info(
            f"Event {event_class} ({EVENT_NAMES[event_class]}): "
            f"frame_idx={frame_idx}, prob={probabilities[frame_idx, event_class].item():.4f}"
        )

    return event_frames


def process_video_for_events(
    model: GolfDBFrameClassifier,
    video_tensor: torch.Tensor,
    n_sequences: int,
    seq_len: int,
) -> dict[int, int]:
    """
    Process video through model and extract event frame indices.

    Args:
        model: Loaded GolfDBFrameClassifier model
        video_tensor: Preprocessed video tensor [N_SEQUENCES, C, SEQ_LEN, H, W]
        n_sequences: Number of sequences
        seq_len: Sequence length

    Returns:
        Dictionary mapping event_class (0-7) to frame_index
    """
    logger.info(f"Running inference on {n_sequences} sequences")

    # Get model predictions
    logits = predict_frame_labels(model, video_tensor, n_sequences, seq_len)

    # Extract event frames
    event_frames = extract_event_frames(logits, n_sequences, seq_len)

    logger.info(f"Extracted {len(event_frames)} event frames")

    return event_frames

