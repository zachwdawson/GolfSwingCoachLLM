"""Video preprocessing utilities for model inference."""
import logging

try:
    import torch
    import torchvision
    from torchvision.transforms.functional import resize
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    torchvision = None  # type: ignore
    resize = None  # type: ignore

logger = logging.getLogger(__name__)


def preprocess_video(video_path: str, target_size: int = 112) -> torch.Tensor:
    """
    Preprocess video file for model inference.

    Args:
        video_path: Path to video file
        target_size: Target frame size (default: 112x112)

    Returns:
        Tensor of shape [T, H, W, C] where T is number of frames,
        H=W=target_size, C=3 (RGB), dtype=uint8
    """
    try:
        logger.info(f"Reading video from {video_path}")
        frames, _, _ = torchvision.io.read_video(video_path)  # uint8 [T, H, W, C]

        logger.info(f"Resizing frames to {target_size}x{target_size}")
        # Resize frames to target size
        frames = resize(frames.permute(0, 3, 1, 2), target_size)
        frames = frames.permute(0, 2, 3, 1)  # Back to [T, H, W, C]

        logger.info(f"Preprocessed video: {frames.shape}")
        return frames
    except Exception as e:
        logger.error(f"Failed to preprocess video {video_path}: {e}")
        raise


def prepare_video_for_inference(
    video: torch.Tensor, seq_len: int, device: str
) -> tuple[torch.Tensor, int]:
    """
    Prepare preprocessed video tensor for model inference.

    Args:
        video: Preprocessed video tensor [T, H, W, C] uint8
        seq_len: Sequence length for model (default: 64)
        device: Device to move tensor to

    Returns:
        Tuple of (batched_video, n_sequences) where:
        - batched_video: [N_SEQUENCES, C, SEQ_LEN, H, W] float32 normalized to [0,1]
        - n_sequences: Number of sequences created
    """
    n_frames = len(video)
    n_sequences = int(n_frames // seq_len)
    n_processed_frames = n_sequences * seq_len

    if n_sequences == 0:
        raise ValueError(f"Video too short: {n_frames} frames < {seq_len} required")

    # Clip video to integer multiple of seq_len
    clipped_video = video[:n_processed_frames]  # [N_FRAMES, H, W, C]

    # Reshape to sequences: [N_SEQUENCES, SEQ_LEN, H, W, C]
    clipped_video = clipped_video.reshape([n_sequences, seq_len, *clipped_video.shape[1:]])

    # Normalize to [0, 1] and convert to float
    clipped_video = clipped_video.float().div_(255.0)

    # Permute to [N_SEQUENCES, C, SEQ_LEN, H, W] for model input
    clipped_video = clipped_video.permute(0, 4, 1, 2, 3)

    # Move to device
    clipped_video = clipped_video.to(device)

    logger.info(
        f"Prepared video: {n_frames} frames -> {n_sequences} sequences of {seq_len} frames"
    )

    return clipped_video, n_sequences

