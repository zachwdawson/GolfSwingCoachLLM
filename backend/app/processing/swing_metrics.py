"""
Swing metrics calculation module for golf swing analysis.

This module computes biomechanical metrics from pose keypoints for four golf swing
positions: address, top, impact, and finish.

Sign Conventions and Assumptions:
- Image coordinates: y-down, x-right (standard image coordinates)
- Normalized coordinates: [0, 1] range
- Target vector: (0, -1) for vertical (straight up/down in image)
- For angle calculations, we convert to math coordinates (y-up) where needed
- Right-handed golfer assumed: lead arm = left arm, trail arm = right arm
- Missing keypoints or low confidence scores return np.nan

MoveNet Keypoint Indices:
- 0: nose (used as neck proxy)
- 5: left_shoulder (lead shoulder)
- 6: right_shoulder (trail shoulder)
- 7: left_elbow (lead elbow)
- 8: right_elbow (trail elbow)
- 9: left_wrist (lead wrist)
- 10: right_wrist (trail wrist)
- 11: left_hip (lead hip)
- 12: right_hip (trail hip)
- 15: left_ankle (lead ankle)
- 16: right_ankle (trail ankle)
"""
import numpy as np
from typing import Dict, Optional, Tuple

# Keypoint indices (MoveNet format)
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_ANKLE = 15
R_ANKLE = 16

# Minimum confidence threshold for keypoints
KEYPOINT_THRESHOLD = 0.11

# Target vector: vertical (straight up/down) in image coordinates (y-down)
# In image coords: (0, -1) means pointing up (negative y)
TARGET_VEC = np.array([0.0, -1.0])


def extract_keypoint(
    keypoints: np.ndarray, idx: int, threshold: float = KEYPOINT_THRESHOLD
) -> Optional[np.ndarray]:
    """
    Extract a keypoint from the keypoints array with validation.

    Args:
        keypoints: [1, 1, 17, 3] numpy array with (y, x, score) format
        idx: Keypoint index (0-16)
        threshold: Minimum confidence score (default: 0.11)

    Returns:
        (x, y) coordinates as numpy array, or None if missing/low confidence
    """
    if keypoints is None or keypoints.shape != (1, 1, 17, 3):
        return None

    kpt = keypoints[0, 0, idx]  # Shape: (3,) with (y, x, score)
    if kpt[2] < threshold:  # score < threshold
        return None

    # Return (x, y) in normalized coordinates [0, 1]
    return np.array([kpt[1], kpt[0]])


def angle_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute angle between two vectors in degrees.

    Args:
        vec1: First vector (2D)
        vec2: Second vector (2D)

    Returns:
        Angle in degrees [0, 180]
    """
    if vec1 is None or vec2 is None:
        return np.nan

    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return np.nan

    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2

    # Compute dot product and clip to [-1, 1] for arccos
    dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def line_angle_deg(
    pt1: np.ndarray, pt2: np.ndarray, target_vec: np.ndarray = TARGET_VEC
) -> float:
    """
    Compute angle of line (pt1 -> pt2) relative to target vector.

    Args:
        pt1: First point (x, y)
        pt2: Second point (x, y)
        target_vec: Target vector (default: vertical up)

    Returns:
        Angle in degrees. Positive = rotated clockwise from target.
    """
    if pt1 is None or pt2 is None:
        return np.nan

    # Line vector: pt2 - pt1
    line_vec = pt2 - pt1

    if np.linalg.norm(line_vec) == 0:
        return np.nan

    # Compute angle between line and target vector
    angle = angle_deg(line_vec, target_vec)

    # Determine sign: cross product to check rotation direction
    # For image coords (y-down), we need to account for coordinate system
    cross = line_vec[0] * target_vec[1] - line_vec[1] * target_vec[0]
    if cross < 0:
        angle = -angle

    return angle


def compute_address_metrics(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for address position.

    Metrics:
    - spine_tilt_deg: Angle between spine line (hip center -> neck) and vertical
    - shoulder_to_target_deg: Angle of shoulder line relative to target
    - pelvis_center_x_norm: Pelvis center x normalized by shoulder width

    Args:
        keypoints: [1, 1, 17, 3] numpy array

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints
    nose = extract_keypoint(keypoints, NOSE)
    l_shoulder = extract_keypoint(keypoints, L_SHOULDER)
    r_shoulder = extract_keypoint(keypoints, R_SHOULDER)
    l_hip = extract_keypoint(keypoints, L_HIP)
    r_hip = extract_keypoint(keypoints, R_HIP)

    # Spine tilt: angle between spine line (hip center -> neck) and vertical
    if l_hip is not None and r_hip is not None and nose is not None:
        hip_center = (l_hip + r_hip) / 2.0
        spine_vec = nose - hip_center
        metrics["spine_tilt_deg"] = angle_deg(spine_vec, TARGET_VEC)
    else:
        metrics["spine_tilt_deg"] = np.nan

    # Shoulder alignment to target
    if l_shoulder is not None and r_shoulder is not None:
        metrics["shoulder_to_target_deg"] = abs(
            line_angle_deg(r_shoulder, l_shoulder, TARGET_VEC)
        )
    else:
        metrics["shoulder_to_target_deg"] = np.nan

    # Pelvis center x normalized by shoulder width
    if l_hip is not None and r_hip is not None:
        pelvis_center = (l_hip + r_hip) / 2.0
        pelvis_center_x = pelvis_center[0]

        if l_shoulder is not None and r_shoulder is not None:
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            if shoulder_width > 0:
                metrics["pelvis_center_x_norm"] = pelvis_center_x / shoulder_width
            else:
                metrics["pelvis_center_x_norm"] = np.nan
        else:
            metrics["pelvis_center_x_norm"] = pelvis_center_x
    else:
        metrics["pelvis_center_x_norm"] = np.nan

    return metrics


def compute_top_metrics(
    keypoints: np.ndarray, address_keypoints: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for top position.

    Metrics:
    - shoulder_turn_deg: Shoulder rotation using foreshortening (width ratio)
    - pelvis_turn_deg: Pelvis rotation using foreshortening
    - x_factor_deg: shoulder_turn - pelvis_turn

    Args:
        keypoints: [1, 1, 17, 3] numpy array for top position
        address_keypoints: [1, 1, 17, 3] numpy array for address position

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints from top position
    l_shoulder_top = extract_keypoint(keypoints, L_SHOULDER)
    r_shoulder_top = extract_keypoint(keypoints, R_SHOULDER)
    l_hip_top = extract_keypoint(keypoints, L_HIP)
    r_hip_top = extract_keypoint(keypoints, R_HIP)

    # Extract baseline widths from address
    l_shoulder_addr = extract_keypoint(address_keypoints, L_SHOULDER)
    r_shoulder_addr = extract_keypoint(address_keypoints, R_SHOULDER)
    l_hip_addr = extract_keypoint(address_keypoints, L_HIP)
    r_hip_addr = extract_keypoint(address_keypoints, R_HIP)

    # Shoulder turn: using foreshortening
    if (
        l_shoulder_top is not None
        and r_shoulder_top is not None
        and l_shoulder_addr is not None
        and r_shoulder_addr is not None
    ):
        w0 = np.linalg.norm(r_shoulder_addr - l_shoulder_addr)
        wt = np.linalg.norm(r_shoulder_top - l_shoulder_top)

        if w0 > 0:
            ratio = np.clip(wt / w0, 0.0, 1.0)
            metrics["shoulder_turn_deg"] = np.degrees(np.arccos(ratio))
        else:
            metrics["shoulder_turn_deg"] = np.nan
    else:
        metrics["shoulder_turn_deg"] = np.nan

    # Pelvis turn: using foreshortening
    if (
        l_hip_top is not None
        and r_hip_top is not None
        and l_hip_addr is not None
        and r_hip_addr is not None
    ):
        h0 = np.linalg.norm(r_hip_addr - l_hip_addr)
        ht = np.linalg.norm(r_hip_top - l_hip_top)

        if h0 > 0:
            ratio = np.clip(ht / h0, 0.0, 1.0)
            metrics["pelvis_turn_deg"] = np.degrees(np.arccos(ratio))
        else:
            metrics["pelvis_turn_deg"] = np.nan
    else:
        metrics["pelvis_turn_deg"] = np.nan

    # X-factor: shoulder_turn - pelvis_turn
    if (
        not np.isnan(metrics["shoulder_turn_deg"])
        and not np.isnan(metrics["pelvis_turn_deg"])
    ):
        metrics["x_factor_deg"] = (
            metrics["shoulder_turn_deg"] - metrics["pelvis_turn_deg"]
        )
    else:
        metrics["x_factor_deg"] = np.nan

    return metrics


def compute_impact_metrics(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for impact position.

    Metrics:
    - hip_open_deg: Hip line angle relative to target
    - shoulder_open_deg: Shoulder line angle relative to target
    - forward_lean_deg: Lead forearm angle (L_ELBOW -> L_WRIST) relative to vertical

    Args:
        keypoints: [1, 1, 17, 3] numpy array

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints
    l_shoulder = extract_keypoint(keypoints, L_SHOULDER)
    r_shoulder = extract_keypoint(keypoints, R_SHOULDER)
    l_hip = extract_keypoint(keypoints, L_HIP)
    r_hip = extract_keypoint(keypoints, R_HIP)
    l_elbow = extract_keypoint(keypoints, L_ELBOW)
    l_wrist = extract_keypoint(keypoints, L_WRIST)

    # Hip open angle to target
    if l_hip is not None and r_hip is not None:
        metrics["hip_open_deg"] = abs(line_angle_deg(r_hip, l_hip, TARGET_VEC))
    else:
        metrics["hip_open_deg"] = np.nan

    # Shoulder open angle to target
    if l_shoulder is not None and r_shoulder is not None:
        metrics["shoulder_open_deg"] = abs(
            line_angle_deg(r_shoulder, l_shoulder, TARGET_VEC)
        )
    else:
        metrics["shoulder_open_deg"] = np.nan

    # Forward shaft lean proxy: lead forearm angle
    if l_elbow is not None and l_wrist is not None:
        forearm_vec = l_wrist - l_elbow
        metrics["forward_lean_deg"] = angle_deg(forearm_vec, TARGET_VEC)
    else:
        metrics["forward_lean_deg"] = np.nan

    return metrics


def compute_finish_metrics(
    keypoints: np.ndarray, address_keypoints: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for finish position.

    Metrics:
    - balance_offset_norm: Pelvis center x offset from lead ankle, normalized
    - shoulder_finish_deg: Shoulder line angle relative to target
    - lead_elbow_angle_deg: Elbow angle at lead arm (shoulder->elbow vs wrist->elbow)

    Args:
        keypoints: [1, 1, 17, 3] numpy array for finish position
        address_keypoints: [1, 1, 17, 3] numpy array for address position

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints from finish position
    l_shoulder = extract_keypoint(keypoints, L_SHOULDER)
    r_shoulder = extract_keypoint(keypoints, R_SHOULDER)
    l_hip = extract_keypoint(keypoints, L_HIP)
    r_hip = extract_keypoint(keypoints, R_HIP)
    l_ankle = extract_keypoint(keypoints, L_ANKLE)
    l_elbow = extract_keypoint(keypoints, L_ELBOW)
    l_wrist = extract_keypoint(keypoints, L_WRIST)

    # Balance offset: pelvis center x offset from lead ankle, normalized
    if l_hip is not None and r_hip is not None and l_ankle is not None:
        pelvis_center = (l_hip + r_hip) / 2.0
        balance_offset = pelvis_center[0] - l_ankle[0]

        # Normalize by shoulder width from address
        l_shoulder_addr = extract_keypoint(address_keypoints, L_SHOULDER)
        r_shoulder_addr = extract_keypoint(address_keypoints, R_SHOULDER)

        if l_shoulder_addr is not None and r_shoulder_addr is not None:
            shoulder_width0 = np.linalg.norm(r_shoulder_addr - l_shoulder_addr)
            if shoulder_width0 > 0:
                metrics["balance_offset_norm"] = balance_offset / shoulder_width0
            else:
                metrics["balance_offset_norm"] = np.nan
        else:
            # Cannot normalize without address shoulder width
            metrics["balance_offset_norm"] = np.nan
    else:
        metrics["balance_offset_norm"] = np.nan

    # Shoulder finish angle
    if l_shoulder is not None and r_shoulder is not None:
        metrics["shoulder_finish_deg"] = abs(
            line_angle_deg(r_shoulder, l_shoulder, TARGET_VEC)
        )
    else:
        metrics["shoulder_finish_deg"] = np.nan

    # Lead arm elbow angle: angle between vectors (shoulder->elbow) and (wrist->elbow)
    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        vec1 = l_elbow - l_shoulder  # shoulder -> elbow
        vec2 = l_elbow - l_wrist  # wrist -> elbow
        metrics["lead_elbow_angle_deg"] = angle_deg(vec1, vec2)
    else:
        metrics["lead_elbow_angle_deg"] = np.nan

    return metrics


def compute_metrics(swing_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute swing metrics for all four positions.

    Args:
        swing_dict: Dictionary with keys "address", "top", "impact", "finish"
                   Each value is a [1, 1, 17, 3] numpy array of keypoints

    Returns:
        Nested dictionary with metrics per position:
        {
            "address": {"spine_tilt_deg": ..., "shoulder_to_target_deg": ..., ...},
            "top": {"shoulder_turn_deg": ..., "pelvis_turn_deg": ..., ...},
            "impact": {"hip_open_deg": ..., "shoulder_open_deg": ..., ...},
            "finish": {"balance_offset_norm": ..., "shoulder_finish_deg": ..., ...}
        }
    """
    result = {}

    # Get address keypoints for baseline measurements
    address_keypoints = swing_dict.get("address")

    # Compute metrics for each position
    if "address" in swing_dict:
        result["address"] = compute_address_metrics(swing_dict["address"])

    if "top" in swing_dict and address_keypoints is not None:
        result["top"] = compute_top_metrics(swing_dict["top"], address_keypoints)
    elif "top" in swing_dict:
        result["top"] = {
            "shoulder_turn_deg": np.nan,
            "pelvis_turn_deg": np.nan,
            "x_factor_deg": np.nan,
        }

    if "impact" in swing_dict:
        result["impact"] = compute_impact_metrics(swing_dict["impact"])

    if "finish" in swing_dict and address_keypoints is not None:
        result["finish"] = compute_finish_metrics(swing_dict["finish"], address_keypoints)
    elif "finish" in swing_dict:
        result["finish"] = {
            "balance_offset_norm": np.nan,
            "shoulder_finish_deg": np.nan,
            "lead_elbow_angle_deg": np.nan,
        }

    return result

