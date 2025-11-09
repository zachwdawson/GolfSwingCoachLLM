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

    Args:
        logits: Model logits [N_FRAMES, N_CLASSES]
        n_sequences: Number of sequences processed
        seq_len: Sequence length

    Returns:
        Dictionary mapping event_class (0-7) to frame_index
    """
    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)  # [N_FRAMES, N_CLASSES]

    # For each event class (0-7), find frame with maximum probability
    event_frames = {}
    for event_class in range(NUM_EVENTS):
        # Get probabilities for this event class across all frames
        class_probs = probabilities[:, event_class]  # [N_FRAMES]

        # Find frame with maximum probability
        frame_idx = torch.argmax(class_probs).item()
        event_frames[event_class] = frame_idx

        logger.debug(
            f"Event {event_class} ({EVENT_NAMES[event_class]}): "
            f"frame {frame_idx}, prob={class_probs[frame_idx].item():.4f}"
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

