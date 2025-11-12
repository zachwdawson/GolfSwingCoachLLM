"""
Swing metrics calculation module for golf swing analysis.

This module computes biomechanical metrics from pose keypoints for four golf swing
positions: address, top, impact, and finish.

Sign Conventions and Assumptions:
- Image coordinates: y-down, x-right (standard image coordinates)
- Normalized coordinates: [0, 1] range
- Target vector: (1, 0) for horizontal target line (left→right in image)
- Vertical vector: (0, -1) for vertical (straight up/down in image)
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
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

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

# Target vector: horizontal target line (left→right) in image coordinates
TARGET_VEC = np.array([1.0, 0.0])

# Vertical vector: vertical (straight up/down) in image coordinates (y-down)
# In image coords: (0, -1) means pointing up (negative y)
VERTICAL_VEC = np.array([0.0, -1.0])


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


def to_math(v_img: np.ndarray) -> np.ndarray:
    """
    Convert image coordinates (y-down) to math coordinates (y-up).

    Args:
        v_img: Vector in image coordinates (x, y) where y increases downward

    Returns:
        Vector in math coordinates (x, y) where y increases upward
    """
    return np.array([v_img[0], -v_img[1]], dtype=float)


def signed_angle_deg(u_img: np.ndarray, v_img: np.ndarray) -> float:
    """
    Compute signed angle between two vectors in degrees.

    Args:
        u_img: First vector in image coordinates (x, y)
        v_img: Second vector in image coordinates (x, y)

    Returns:
        Signed angle in degrees. Positive = counterclockwise from u to v.
    """
    if u_img is None or v_img is None:
        return np.nan

    u, v = to_math(u_img), to_math(v_img)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)

    if nu == 0 or nv == 0:
        return np.nan

    u, v = u / nu, v / nv
    cross = u[0] * v[1] - u[1] * v[0]
    dot = np.clip(u[0] * v[0] + u[1] * v[1], -1.0, 1.0)

    return float(np.degrees(np.arctan2(cross, dot)))


def angle_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute unsigned angle between two vectors in degrees.

    Args:
        vec1: First vector (2D) in image coordinates
        vec2: Second vector (2D) in image coordinates

    Returns:
        Angle in degrees [0, 180]
    """
    return abs(signed_angle_deg(vec1, vec2))


def acute_angle_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute acute angle between two vectors in degrees.
    Returns the smaller angle (0-90°) for line-to-line comparisons.

    Args:
        vec1: First vector (2D) in image coordinates
        vec2: Second vector (2D) in image coordinates

    Returns:
        Acute angle in degrees [0, 90]
    """
    angle = abs(signed_angle_deg(vec1, vec2))
    # Normalize to acute angle: if > 90°, use 180° - angle
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def line_vec(p: np.ndarray, q: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute line vector from point p to point q.

    Args:
        p: First point (x, y)
        q: Second point (x, y)

    Returns:
        Line vector (q - p), or None if invalid
    """
    if p is None or q is None:
        return None

    v = q - p
    return v if np.linalg.norm(v) > 0 else None


def width(p_left: np.ndarray, p_right: np.ndarray) -> Optional[float]:
    """
    Compute width (distance) between two points.

    Args:
        p_left: Left point (x, y)
        p_right: Right point (x, y)

    Returns:
        Distance as float, or None if invalid
    """
    if p_left is None or p_right is None:
        return None

    return float(np.linalg.norm(p_left - p_right))


def horizontal_width(p1: np.ndarray, p2: np.ndarray) -> Optional[float]:
    """
    Compute horizontal width (x-coordinate difference) between two points.
    Used for foreshortening calculations where we measure how horizontal width
    decreases when rotating.
    
    This function computes the absolute difference in x-coordinates,
    which represents the horizontal span between the two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Horizontal width (absolute x-difference) as float, or None if invalid
    """
    if p1 is None or p2 is None:
        return None

    return float(abs(p2[0] - p1[0]))


def line_angle_deg(
    pt1: np.ndarray, pt2: np.ndarray, target_vec: np.ndarray = TARGET_VEC
) -> float:
    """
    Compute angle of line (pt1 -> pt2) relative to target vector.
    Returns the acute angle (0-90°) for line-to-line comparisons.

    Args:
        pt1: First point (x, y)
        pt2: Second point (x, y)
        target_vec: Target vector (default: horizontal target line)

    Returns:
        Acute angle in degrees [0, 90]
    """
    line_v = line_vec(pt1, pt2)
    if line_v is None:
        return np.nan

    return acute_angle_deg(line_v, target_vec)


def compute_address_metrics(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for address position.

    Metrics:
    - spine_tilt_deg: Angle between spine line (hip center -> neck) and vertical
    - shoulder_to_target_deg: Angle of shoulder line relative to target line
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

    # Use mid-shoulders as neck proxy, fallback to nose if shoulders missing
    if l_shoulder is not None and r_shoulder is not None:
        neck = (l_shoulder + r_shoulder) / 2.0
    else:
        neck = nose

    # Spine tilt: angle between spine line (hip center -> neck) and vertical
    if l_hip is not None and r_hip is not None and neck is not None:
        hip_center = (l_hip + r_hip) / 2.0
        spine_v = line_vec(hip_center, neck)
        if spine_v is not None:
            metrics["spine_tilt_deg"] = acute_angle_deg(spine_v, VERTICAL_VEC)
        else:
            metrics["spine_tilt_deg"] = np.nan
    else:
        metrics["spine_tilt_deg"] = np.nan

    # Shoulder alignment to target line
    if l_shoulder is not None and r_shoulder is not None:
        metrics["shoulder_to_target_deg"] = line_angle_deg(
            r_shoulder, l_shoulder, TARGET_VEC
        )
    else:
        metrics["shoulder_to_target_deg"] = np.nan

    # Pelvis center x normalized by shoulder width
    if l_hip is not None and r_hip is not None:
        pelvis_center = (l_hip + r_hip) / 2.0
        pelvis_center_x = pelvis_center[0]

        if l_shoulder is not None and r_shoulder is not None:
            shoulder_width = width(l_shoulder, r_shoulder)
            if shoulder_width is not None and shoulder_width > 0:
                metrics["pelvis_center_x_norm"] = pelvis_center_x / shoulder_width
            else:
                metrics["pelvis_center_x_norm"] = np.nan
        else:
            metrics["pelvis_center_x_norm"] = np.nan
    else:
        metrics["pelvis_center_x_norm"] = np.nan

    return metrics


def compute_top_metrics(
    keypoints: np.ndarray, address_keypoints: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for top position.

    Metrics:
    - shoulder_turn_deg: Shoulder rotation angle computed from change in shoulder line orientation
    - pelvis_turn_deg: Pelvis rotation angle computed from change in pelvis line orientation
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

    # Shoulder turn: compute rotation angle from change in shoulder line orientation
    # Compute shoulder line vectors: from left shoulder to right shoulder
    if (l_shoulder_addr is not None and r_shoulder_addr is not None and
        l_shoulder_top is not None and r_shoulder_top is not None):
        # Shoulder line vector at address: right shoulder - left shoulder
        shoulder_vec_addr = r_shoulder_addr - l_shoulder_addr
        # Shoulder line vector at top: right shoulder - left shoulder
        shoulder_vec_top = r_shoulder_top - l_shoulder_top
        
        # Calculate angle between the two vectors
        # This gives the rotation angle of the shoulders
        metrics["shoulder_turn_deg"] = angle_deg(shoulder_vec_addr, shoulder_vec_top)
        
        logger.info(
            f"Top metrics - Shoulder: turn_deg={metrics['shoulder_turn_deg']:.2f}"
        )
        logger.info(
            f"  Address shoulders: L={l_shoulder_addr[0]:.4f}, R={r_shoulder_addr[0]:.4f}, "
            f"vector=({shoulder_vec_addr[0]:.4f}, {shoulder_vec_addr[1]:.4f})"
        )
        logger.info(
            f"  Top shoulders: L={l_shoulder_top[0]:.4f}, R={r_shoulder_top[0]:.4f}, "
            f"vector=({shoulder_vec_top[0]:.4f}, {shoulder_vec_top[1]:.4f})"
        )
    else:
        metrics["shoulder_turn_deg"] = np.nan

    # Pelvis turn: compute rotation angle from change in pelvis line orientation
    # Compute pelvis line vectors: from left hip to right hip
    if (l_hip_addr is not None and r_hip_addr is not None and
        l_hip_top is not None and r_hip_top is not None):
        # Pelvis line vector at address: right hip - left hip
        pelvis_vec_addr = r_hip_addr - l_hip_addr
        # Pelvis line vector at top: right hip - left hip
        pelvis_vec_top = r_hip_top - l_hip_top
        
        # Calculate angle between the two vectors
        # This gives the rotation angle of the pelvis
        metrics["pelvis_turn_deg"] = angle_deg(pelvis_vec_addr, pelvis_vec_top)
        
        logger.info(
            f"Top metrics - Pelvis: turn_deg={metrics['pelvis_turn_deg']:.2f}"
        )
        logger.info(
            f"  Address hips: L={l_hip_addr[0]:.4f}, R={r_hip_addr[0]:.4f}, "
            f"vector=({pelvis_vec_addr[0]:.4f}, {pelvis_vec_addr[1]:.4f})"
        )
        logger.info(
            f"  Top hips: L={l_hip_top[0]:.4f}, R={r_hip_top[0]:.4f}, "
            f"vector=({pelvis_vec_top[0]:.4f}, {pelvis_vec_top[1]:.4f})"
        )
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


def compute_impact_metrics(
    keypoints: np.ndarray, handedness: str = "right"
) -> Dict[str, float]:
    """
    Compute metrics for impact position.

    Metrics:
    - hip_open_deg: Hip line angle relative to target
    - shoulder_open_deg: Shoulder line angle relative to target
    - forward_lean_deg: Lead forearm angle relative to vertical

    Args:
        keypoints: [1, 1, 17, 3] numpy array
        handedness: "right" for right-handed (default), "left" for left-handed

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints
    l_shoulder = extract_keypoint(keypoints, L_SHOULDER)
    r_shoulder = extract_keypoint(keypoints, R_SHOULDER)
    l_hip = extract_keypoint(keypoints, L_HIP)
    r_hip = extract_keypoint(keypoints, R_HIP)
    
    # Get lead arm keypoints based on handedness
    if handedness == "left":
        lead_elbow = extract_keypoint(keypoints, R_ELBOW)
        lead_wrist = extract_keypoint(keypoints, R_WRIST)
    else:  # right-handed (default)
        lead_elbow = extract_keypoint(keypoints, L_ELBOW)
        lead_wrist = extract_keypoint(keypoints, L_WRIST)

    # Hip open angle to target line
    if l_hip is not None and r_hip is not None:
        metrics["hip_open_deg"] = line_angle_deg(r_hip, l_hip, TARGET_VEC)
    else:
        metrics["hip_open_deg"] = np.nan

    # Shoulder open angle to target line
    if l_shoulder is not None and r_shoulder is not None:
        metrics["shoulder_open_deg"] = line_angle_deg(
            r_shoulder, l_shoulder, TARGET_VEC
        )
    else:
        metrics["shoulder_open_deg"] = np.nan

    # Forward shaft lean proxy: lead forearm angle relative to vertical
    if lead_elbow is not None and lead_wrist is not None:
        forearm_v = line_vec(lead_elbow, lead_wrist)
        if forearm_v is not None:
            metrics["forward_lean_deg"] = acute_angle_deg(forearm_v, VERTICAL_VEC)
        else:
            metrics["forward_lean_deg"] = np.nan
    else:
        metrics["forward_lean_deg"] = np.nan

    return metrics


def compute_finish_metrics(
    keypoints: np.ndarray, address_keypoints: np.ndarray, handedness: str = "right"
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
        handedness: "right" for right-handed (default), "left" for left-handed

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints from finish position
    l_shoulder = extract_keypoint(keypoints, L_SHOULDER)
    r_shoulder = extract_keypoint(keypoints, R_SHOULDER)
    l_hip = extract_keypoint(keypoints, L_HIP)
    r_hip = extract_keypoint(keypoints, R_HIP)
    
    # Get lead body parts based on handedness
    if handedness == "left":
        lead_ankle = extract_keypoint(keypoints, R_ANKLE)
        lead_shoulder = extract_keypoint(keypoints, R_SHOULDER)
        lead_elbow = extract_keypoint(keypoints, R_ELBOW)
        lead_wrist = extract_keypoint(keypoints, R_WRIST)
    else:  # right-handed (default)
        lead_ankle = extract_keypoint(keypoints, L_ANKLE)
        lead_shoulder = extract_keypoint(keypoints, L_SHOULDER)
        lead_elbow = extract_keypoint(keypoints, L_ELBOW)
        lead_wrist = extract_keypoint(keypoints, L_WRIST)

    # Balance offset: pelvis center x offset from lead ankle, normalized
    if l_hip is not None and r_hip is not None and lead_ankle is not None:
        pelvis_center = (l_hip + r_hip) / 2.0
        balance_offset = pelvis_center[0] - lead_ankle[0]

        # Normalize by shoulder width from address
        l_shoulder_addr = extract_keypoint(address_keypoints, L_SHOULDER)
        r_shoulder_addr = extract_keypoint(address_keypoints, R_SHOULDER)

        if l_shoulder_addr is not None and r_shoulder_addr is not None:
            shoulder_width0 = width(l_shoulder_addr, r_shoulder_addr)
            if shoulder_width0 is not None and shoulder_width0 > 0:
                metrics["balance_offset_norm"] = balance_offset / shoulder_width0
            else:
                metrics["balance_offset_norm"] = np.nan
        else:
            # Cannot normalize without address shoulder width
            metrics["balance_offset_norm"] = np.nan
    else:
        metrics["balance_offset_norm"] = np.nan

    # Shoulder finish angle to target line
    if l_shoulder is not None and r_shoulder is not None:
        metrics["shoulder_finish_deg"] = line_angle_deg(
            r_shoulder, l_shoulder, TARGET_VEC
        )
    else:
        metrics["shoulder_finish_deg"] = np.nan

    # Lead arm elbow angle: angle between vectors (shoulder->elbow) and (wrist->elbow)
    if lead_shoulder is not None and lead_elbow is not None and lead_wrist is not None:
        vec1 = lead_elbow - lead_shoulder  # shoulder -> elbow
        vec2 = lead_elbow - lead_wrist  # wrist -> elbow
        metrics["lead_elbow_angle_deg"] = angle_deg(vec1, vec2)
    else:
        metrics["lead_elbow_angle_deg"] = np.nan

    return metrics


def compute_metrics(
    swing_dict: Dict[str, np.ndarray], handedness: str = "right"
) -> Dict[str, Dict[str, float]]:
    """
    Compute swing metrics for all four positions.

    Args:
        swing_dict: Dictionary with keys "address", "top", "impact", "finish"
                   Each value is a [1, 1, 17, 3] numpy array of keypoints
        handedness: "right" for right-handed (default), "left" for left-handed

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
        result["impact"] = compute_impact_metrics(swing_dict["impact"], handedness)

    if "finish" in swing_dict and address_keypoints is not None:
        result["finish"] = compute_finish_metrics(
            swing_dict["finish"], address_keypoints, handedness
        )
    elif "finish" in swing_dict:
        result["finish"] = {
            "balance_offset_norm": np.nan,
            "shoulder_finish_deg": np.nan,
            "lead_elbow_angle_deg": np.nan,
        }

    return result

