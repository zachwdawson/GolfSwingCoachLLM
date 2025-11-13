"""
Pose estimation module using MoveNet from TensorFlow Hub.

This module provides functions to load MoveNet models, estimate poses,
and draw pose overlays on images.
"""
import logging
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from typing import Optional, Tuple, Callable
from PIL import Image

logger = logging.getLogger(__name__)

# Skeleton connections for 17 keypoints (MoveNet format)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# Global variable to cache the loaded model
_movenet_model = None
_movenet_input_size = None


def load_movenet_model(model_name: str = "movenet_thunder") -> Tuple[Callable, int]:
    """
    Load MoveNet model from TensorFlow Hub.

    Args:
        model_name: Model name - "movenet_lightning" or "movenet_thunder"

    Returns:
        Tuple of (movenet_function, input_size)
    """
    global _movenet_model, _movenet_input_size

    # Return cached model if already loaded
    if _movenet_model is not None and _movenet_input_size is not None:
        return _movenet_model, _movenet_input_size

    if "movenet_lightning" in model_name:
        model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        input_size = 192
    elif "movenet_thunder" in model_name:
        model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        input_size = 256
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    try:
        logger.info(f"Loading MoveNet model: {model_name} from {model_url}")
        module = hub.load(model_url)
        _movenet_input_size = input_size

        def movenet(input_image: tf.Tensor) -> np.ndarray:
            """
            Runs pose detection on an input image.

            Args:
                input_image: A [1, height, width, 3] tensor representing the input image
                    pixels. Note that the height/width should already be resized and match the
                    expected input resolution of the model before passing into this function.

            Returns:
                A [1, 1, 17, 3] float numpy array representing the predicted keypoint
                coordinates and scores. Format: [batch, person, keypoint, (y, x, score)]
            """
            model = module.signatures['serving_default']
            # SavedModel format expects tensor type of int32
            input_image = tf.cast(input_image, dtype=tf.int32)
            # Run model inference
            outputs = model(input_image)
            # Output is a [1, 1, 17, 3] tensor
            keypoints_with_scores = outputs['output_0'].numpy()
            return keypoints_with_scores

        _movenet_model = movenet
        logger.info(f"MoveNet model loaded successfully. Input size: {input_size}x{input_size}")
        return movenet, input_size

    except Exception as e:
        logger.error(f"Failed to load MoveNet model: {e}", exc_info=True)
        raise


def estimate_pose(
    image: np.ndarray | Image.Image,
    model_name: str = "movenet_thunder",
    input_size: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate pose keypoints from an image using MoveNet.

    Args:
        image: Input image as numpy array (HxWx3 RGB) or PIL Image
        model_name: Model name - "movenet_lightning" or "movenet_thunder"
        input_size: Override input size (if None, uses model default)

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores. Format: [batch, person, keypoint, (y, x, score)]
        Coordinates are normalized to [0, 1] range.
    """
    # Load model if not already loaded
    movenet_fn, default_input_size = load_movenet_model(model_name)
    if input_size is None:
        input_size = default_input_size

    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        # Convert RGBA to RGB if needed
        if image.shape[2] == 4:
            image = image[:, :, :3]

    # Ensure image is RGB uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Resize and pad image to model input size
    image_tensor = tf.expand_dims(tf.constant(image), axis=0)
    image_tensor = tf.image.resize_with_pad(image_tensor, input_size, input_size)

    # Run pose estimation
    keypoints = movenet_fn(image_tensor)
    return keypoints


def draw_pose_overlay(
    image: np.ndarray | Image.Image,
    keypoints: np.ndarray,
    keypoint_threshold: float = 0.11,
) -> np.ndarray:
    """
    Draw pose overlay (skeleton and keypoints) on an image.

    Args:
        image: Input image as numpy array (HxWx3 RGB) or PIL Image
        keypoints: Keypoints array [1, 1, 17, 3] with (y, x, score) format
        keypoint_threshold: Minimum confidence threshold for drawing keypoints

    Returns:
        Annotated image as numpy array (HxWx3 RGB uint8)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        # Convert RGBA to RGB if needed
        if image.shape[2] == 4:
            image = image[:, :, :3]

    # Ensure image is RGB uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    out = image.copy()
    h, w = out.shape[:2]

    # Extract keypoints: [1, 1, 17, 3] -> [17, 3]
    kpts = keypoints[0, 0]  # Shape: (17, 3)

    # Convert normalized coordinates to pixel coordinates
    # keypoints format: (y, x, score) where y, x are in [0, 1]
    pts = np.stack([w * kpts[:, 1], h * kpts[:, 0]], axis=-1).astype(np.int32)
    conf = kpts[:, 2]

    # Draw skeleton lines (cyan) - much smaller
    for p, q in SKELETON:
        if conf[p] > keypoint_threshold and conf[q] > keypoint_threshold:
            cv2.line(
                out,
                tuple(pts[p]),
                tuple(pts[q]),
                (0, 255, 255),  # Cyan color (BGR format)
                1,  # Reduced from 2 to 1
                lineType=cv2.LINE_AA,
            )

    # Draw keypoint circles (magenta with white border) - much smaller
    for i, (x, y) in enumerate(pts):
        if conf[i] > keypoint_threshold:
            # Draw filled circle (magenta) - much smaller
            cv2.circle(out, (x, y), 1, (255, 20, 147), -1, lineType=cv2.LINE_AA)
            # Draw border circle (white) - much smaller
            cv2.circle(out, (x, y), 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return out

