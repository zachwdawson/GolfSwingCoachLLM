"""
Swing metrics calculation module for golf swing analysis.

This module computes biomechanical metrics from pose keypoints for five golf swing
positions: address, top, mid-downswing, impact, and finish.

Sign Conventions and Assumptions:
- Image coordinates: y-down, x-right (standard image coordinates)
- Normalized coordinates: [0, 1] range
- Target vector: derived per swing from address ankle alignment; fallback is horizontal (left→right)
- Vertical vector: (0, -1) for vertical (straight up/down in image)
- For angle calculations, we convert to math coordinates (y-up) where needed
- Right-handed golfer assumed: lead arm = left arm, trail arm = right arm
- Missing keypoints or low confidence scores return np.nan

Notation:
- Vector(a→b) = b − a where points are (x,y) with y down
- Angle(u,v) = smallest angle between vectors u and v, in degrees
- TARGET = (1,0), VERTICAL = (0,−1)
- Mid(p,q) = (p+q)/2
- Use neck = Mid(L_shoulder, R_shoulder) if both shoulders exist; else use nose

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

# Default target vector: horizontal target line (left→right) in image coordinates
DEFAULT_TARGET_VEC = np.array([1.0, 0.0])

# Vertical vector: vertical (straight up/down) in image coordinates (y-down)
# In image coords: (0, -1) means pointing up (negative y)
VERTICAL_VEC = np.array([0.0, -1.0])

# Assumed finish-facing angles (degrees) for finish-anchored rotation calculation
SHOULDER_FINISH_DEG = 90.0
PELVIS_FINISH_DEG = 85.0


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


def wrap_angle_deg(angle_deg: float) -> float:
    """
    Wrap angle to (-180°, 180°] range.

    Args:
        angle_deg: Angle in degrees

    Returns:
        Wrapped angle in degrees, or np.nan if input is invalid
    """
    if np.isnan(angle_deg):
        return np.nan

    angle_rad = np.radians(angle_deg)
    wrapped_rad = np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
    return float(np.degrees(wrapped_rad))


def line_angle_to_ankles(
    l_body: Optional[np.ndarray],
    r_body: Optional[np.ndarray],
    l_ankle: Optional[np.ndarray],
    r_ankle: Optional[np.ndarray],
) -> float:
    """
    Compute signed angle from ankle line to body line (shoulder or hip).

    The angle is measured from the ankle line (target baseline) to the body line.
    Positive values indicate counterclockwise rotation from ankle line to body line.

    Args:
        l_body: Left body point (shoulder or hip) (x, y)
        r_body: Right body point (shoulder or hip) (x, y)
        l_ankle: Left ankle point (x, y)
        r_ankle: Right ankle point (x, y)

    Returns:
        Signed angle in degrees from ankle line to body line, or np.nan if invalid
    """
    # Build vectors: ankle line (right→left) and body line (right→left)
    ankle_vec = line_vec(r_ankle, l_ankle)  # right→left ankle (target baseline)
    body_vec = line_vec(r_body, l_body)    # right→left body

    if ankle_vec is None or body_vec is None:
        return np.nan

    # Compute signed angle from ankle line to body line
    # This gives the rotation of the body line relative to the ankle line
    return signed_angle_deg(ankle_vec, body_vec)


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


def line_angle_deg(pt1: np.ndarray, pt2: np.ndarray, target_vec: np.ndarray) -> float:
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


def safe_arccos_ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    """
    Return arccos(clip(numer/denom, 0, 1)) in radians, or None if bad/degenerate.

    Args:
        numer: Numerator value
        denom: Denominator value

    Returns:
        Arccos value in radians, or None if invalid
    """
    if denom is None or numer is None:
        return None
    if denom <= 0:
        return None
    r = max(0.0, min(1.0, numer / denom))
    return float(np.arccos(r))


def compute_target_vector(
    address_keypoints: Optional[np.ndarray], handedness: str = "right"
) -> np.ndarray:
    """
    Derive a swing-specific target vector from address ankle keypoints.

    Args:
        address_keypoints: [1, 1, 17, 3] numpy array for address position.
        handedness: "right" for right-handed (default), "left" for left-handed.

    Returns:
        Normalized 2D vector representing the target line direction, oriented to align
        with the default positive x direction when possible. Falls back to
        DEFAULT_TARGET_VEC if ankle data is unavailable or degenerate.
    """
    if address_keypoints is None:
        logger.debug("Address keypoints missing; using default target vector.")
        return DEFAULT_TARGET_VEC

    if handedness == "left":
        lead_idx, trail_idx = R_ANKLE, L_ANKLE
    else:
        lead_idx, trail_idx = L_ANKLE, R_ANKLE

    lead_ankle = extract_keypoint(address_keypoints, lead_idx)
    trail_ankle = extract_keypoint(address_keypoints, trail_idx)
    target_vec = line_vec(trail_ankle, lead_ankle)

    if target_vec is None:
        logger.debug(
            "Unable to derive target vector from ankles; using default target vector."
        )
        return DEFAULT_TARGET_VEC

    norm = np.linalg.norm(target_vec)
    if norm < 1e-6:
        logger.debug(
            "Derived target vector magnitude too small; using default target vector."
        )
        return DEFAULT_TARGET_VEC

    target_vec = target_vec / norm

    # Ensure consistent orientation with default target direction.
    if np.dot(target_vec, DEFAULT_TARGET_VEC) < 0:
        target_vec = -target_vec

    return target_vec


def compute_address_metrics(
    keypoints: np.ndarray, target_vec: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for address position (P1).

    Metrics:
    - spine_forward_bend_deg: Spine forward bend (20-40°)
      Compute: hip_c = Mid(L_hip, R_hip); spine = Vector(hip_c→neck); Angle(spine, VERTICAL)
    - shoulder_alignment_deg: Shoulder alignment to target (0-10°)
      Compute: sh_line = Vector(R_shoulder→L_shoulder); Angle(sh_line, target_vec)

    Args:
        keypoints: [1, 1, 17, 3] numpy array
        target_vec: Target vector for rotation baseline

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

    # Spine forward bend: hip_c = Mid(L_hip, R_hip); spine = Vector(hip_c→neck); Angle(spine, VERTICAL)
    if l_hip is not None and r_hip is not None and neck is not None:
        hip_c = (l_hip + r_hip) / 2.0
        spine = line_vec(hip_c, neck)
        if spine is not None:
            metrics["spine_forward_bend_deg"] = acute_angle_deg(spine, VERTICAL_VEC)
        else:
            metrics["spine_forward_bend_deg"] = np.nan
    else:
        metrics["spine_forward_bend_deg"] = np.nan

    # Shoulder alignment to target: sh_line = Vector(R_shoulder→L_shoulder); Angle(sh_line, target_vec)
    if l_shoulder is not None and r_shoulder is not None:
        sh_line = line_vec(r_shoulder, l_shoulder)
        if sh_line is not None:
            metrics["shoulder_alignment_deg"] = acute_angle_deg(sh_line, target_vec)
        else:
            metrics["shoulder_alignment_deg"] = np.nan
    else:
        metrics["shoulder_alignment_deg"] = np.nan

    return metrics


def compute_top_metrics(
    keypoints: np.ndarray,
    address_keypoints: np.ndarray,
    handedness: str = "right",
    finish_keypoints: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute metrics for top position (P4) using ankle-referenced rotation geometry.

    Metrics:
    - shoulder_turn_deg: Shoulder turn (70-100°)
      Compute: Signed angle change from address to top, measured relative to ankle line
    - pelvis_turn_deg: Pelvis turn (25-45°)
      Compute: Signed angle change from address to top, measured relative to ankle line

    The rotation is computed as the signed change in angle between the body line
    (shoulder or hip) and the ankle line (target baseline). The ankle line from
    address is always used as the reference baseline.

    Args:
        keypoints: [1, 1, 17, 3] numpy array for top position
        address_keypoints: [1, 1, 17, 3] numpy array for address position
        handedness: "right" for right-handed (default), "left" for left-handed
        finish_keypoints: [1, 1, 17, 3] numpy array for finish position (deprecated, kept for compatibility)

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints from top position
    l_shoulder_top = extract_keypoint(keypoints, L_SHOULDER)
    r_shoulder_top = extract_keypoint(keypoints, R_SHOULDER)
    l_hip_top = extract_keypoint(keypoints, L_HIP)
    r_hip_top = extract_keypoint(keypoints, R_HIP)

    # Extract keypoints from address position
    l_shoulder_addr = extract_keypoint(address_keypoints, L_SHOULDER)
    r_shoulder_addr = extract_keypoint(address_keypoints, R_SHOULDER)
    l_hip_addr = extract_keypoint(address_keypoints, L_HIP)
    r_hip_addr = extract_keypoint(address_keypoints, R_HIP)

    # Always use address ankles as baseline (per user specification)
    if handedness == "left":
        lead_ankle_idx, trail_ankle_idx = R_ANKLE, L_ANKLE
    else:
        lead_ankle_idx, trail_ankle_idx = L_ANKLE, R_ANKLE

    l_ankle_addr = extract_keypoint(address_keypoints, L_ANKLE)
    r_ankle_addr = extract_keypoint(address_keypoints, R_ANKLE)

    # Compute angles at address: angle from ankle line to body line
    theta_shoulder_addr = line_angle_to_ankles(
        l_shoulder_addr, r_shoulder_addr, l_ankle_addr, r_ankle_addr
    )
    theta_hip_addr = line_angle_to_ankles(
        l_hip_addr, r_hip_addr, l_ankle_addr, r_ankle_addr
    )

    # Compute angles at top: angle from ankle line (address baseline) to body line
    theta_shoulder_top = line_angle_to_ankles(
        l_shoulder_top, r_shoulder_top, l_ankle_addr, r_ankle_addr
    )
    theta_hip_top = line_angle_to_ankles(
        l_hip_top, r_hip_top, l_ankle_addr, r_ankle_addr
    )

    # Compute signed rotation as wrapped difference
    if not np.isnan(theta_shoulder_addr) and not np.isnan(theta_shoulder_top):
        delta_theta_shoulder = wrap_angle_deg(theta_shoulder_top - theta_shoulder_addr)
        metrics["shoulder_turn_deg"] = delta_theta_shoulder
    else:
        metrics["shoulder_turn_deg"] = np.nan

    if not np.isnan(theta_hip_addr) and not np.isnan(theta_hip_top):
        delta_theta_hip = wrap_angle_deg(theta_hip_top - theta_hip_addr)
        metrics["pelvis_turn_deg"] = delta_theta_hip
    else:
        metrics["pelvis_turn_deg"] = np.nan

    return metrics


def compute_mid_downswing_metrics(
    keypoints: np.ndarray, target_vec: np.ndarray, handedness: str = "right"
) -> Dict[str, float]:
    """
    Compute metrics for mid-downswing position (P6).

    Metrics:
    - hip_open_deg: Hip open angle (15-30°)
      Compute: hip_line = Vector(R_hip→L_hip) at P6; Angle(hip_line, target_vec)
    - trail_elbow_flexion_deg: Trail elbow flexion (50-80°)
      Compute (right-handed): v1 = Vector(R_shoulder→R_elbow), v2 = Vector(R_wrist→R_elbow); Angle(v1,v2)

    Args:
        keypoints: [1, 1, 17, 3] numpy array for mid-downswing position
        target_vec: Target vector for rotation baseline
        handedness: "right" for right-handed (default), "left" for left-handed

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints
    l_hip = extract_keypoint(keypoints, L_HIP)
    r_hip = extract_keypoint(keypoints, R_HIP)
    
    # Get trail arm keypoints based on handedness
    if handedness == "left":
        trail_shoulder = extract_keypoint(keypoints, L_SHOULDER)
        trail_elbow = extract_keypoint(keypoints, L_ELBOW)
        trail_wrist = extract_keypoint(keypoints, L_WRIST)
    else:  # right-handed (default)
        trail_shoulder = extract_keypoint(keypoints, R_SHOULDER)
        trail_elbow = extract_keypoint(keypoints, R_ELBOW)
        trail_wrist = extract_keypoint(keypoints, R_WRIST)

    # Hip open angle: hip_line = Vector(R_hip→L_hip); Angle(hip_line, target_vec)
    if l_hip is not None and r_hip is not None:
        hip_line = line_vec(r_hip, l_hip)
        if hip_line is not None:
            metrics["hip_open_deg"] = acute_angle_deg(hip_line, target_vec)
        else:
            metrics["hip_open_deg"] = np.nan
    else:
        metrics["hip_open_deg"] = np.nan

    # Trail elbow flexion: v1 = Vector(trail_shoulder→trail_elbow), v2 = Vector(trail_wrist→trail_elbow); Angle(v1,v2)
    if trail_shoulder is not None and trail_elbow is not None and trail_wrist is not None:
        v1 = line_vec(trail_shoulder, trail_elbow)  # shoulder -> elbow
        v2 = line_vec(trail_wrist, trail_elbow)      # wrist -> elbow
        if v1 is not None and v2 is not None:
            metrics["trail_elbow_flexion_deg"] = angle_deg(v1, v2)
        else:
            metrics["trail_elbow_flexion_deg"] = np.nan
    else:
        metrics["trail_elbow_flexion_deg"] = np.nan

    return metrics


def compute_impact_metrics(
    keypoints: np.ndarray, target_vec: np.ndarray, handedness: str = "right"
) -> Dict[str, float]:
    """
    Compute metrics for impact position (P7).

    Metrics:
    - hip_open_deg: Hip open angle (25-45°)
      Compute: hip_line at impact; Angle(hip_line, target_vec)
    - forward_shaft_lean_deg: Forward shaft-lean proxy (8-20°)
      Compute (right-handed): forearm = Vector(L_elbow→L_wrist); Angle(forearm, VERTICAL)

    Args:
        keypoints: [1, 1, 17, 3] numpy array
        target_vec: Target vector for rotation baseline
        handedness: "right" for right-handed (default), "left" for left-handed

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Extract keypoints
    l_hip = extract_keypoint(keypoints, L_HIP)
    r_hip = extract_keypoint(keypoints, R_HIP)
    
    # Get lead arm keypoints based on handedness
    if handedness == "left":
        lead_elbow = extract_keypoint(keypoints, R_ELBOW)
        lead_wrist = extract_keypoint(keypoints, R_WRIST)
    else:  # right-handed (default)
        lead_elbow = extract_keypoint(keypoints, L_ELBOW)
        lead_wrist = extract_keypoint(keypoints, L_WRIST)

    # Hip open angle: hip_line at impact; Angle(hip_line, target_vec)
    if l_hip is not None and r_hip is not None:
        hip_line = line_vec(r_hip, l_hip)
        if hip_line is not None:
            metrics["hip_open_deg"] = acute_angle_deg(hip_line, target_vec)
        else:
            metrics["hip_open_deg"] = np.nan
    else:
        metrics["hip_open_deg"] = np.nan

    # Forward shaft-lean proxy: forearm = Vector(lead_elbow→lead_wrist); Angle(forearm, VERTICAL)
    if lead_elbow is not None and lead_wrist is not None:
        forearm = line_vec(lead_elbow, lead_wrist)
        if forearm is not None:
            metrics["forward_shaft_lean_deg"] = acute_angle_deg(forearm, VERTICAL_VEC)
        else:
            metrics["forward_shaft_lean_deg"] = np.nan
    else:
        metrics["forward_shaft_lean_deg"] = np.nan

    return metrics


def compute_finish_metrics(
    keypoints: np.ndarray,
    address_keypoints: np.ndarray,
    target_vec: np.ndarray,
    handedness: str = "right",
) -> Dict[str, float]:
    """
    Compute metrics for finish position (P10).

    Metrics:
    - balance_over_lead_foot_norm: Balance over lead foot (norm.) (0.10-0.40)
      Compute: hip_c = Mid(L_hip,R_hip), lead_ankle = L_ankle (RH golfer), sw0 = |R_shoulder−L_shoulder| at address.
               Metric = (hip_c.x − lead_ankle.x)/sw0
    - shoulder_finish_deg: Shoulder finish angle (60-100°)
      Compute: sh_line at finish; Angle(sh_line, target_vec)

    Args:
        keypoints: [1, 1, 17, 3] numpy array for finish position
        address_keypoints: [1, 1, 17, 3] numpy array for address position
        target_vec: Target vector for rotation baseline
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
    else:  # right-handed (default)
        lead_ankle = extract_keypoint(keypoints, L_ANKLE)

    # Balance over lead foot: hip_c = Mid(L_hip,R_hip), lead_ankle = L_ankle (RH golfer), sw0 = |R_shoulder−L_shoulder| at address.
    # Metric = (hip_c.x − lead_ankle.x)/sw0
    if l_hip is not None and r_hip is not None and lead_ankle is not None:
        hip_c = (l_hip + r_hip) / 2.0

        # Normalize by shoulder width from address
        l_shoulder_addr = extract_keypoint(address_keypoints, L_SHOULDER)
        r_shoulder_addr = extract_keypoint(address_keypoints, R_SHOULDER)

        if l_shoulder_addr is not None and r_shoulder_addr is not None:
            sw0 = width(l_shoulder_addr, r_shoulder_addr)
            if sw0 is not None and sw0 > 0:
                metrics["balance_over_lead_foot_norm"] = (hip_c[0] - lead_ankle[0]) / sw0
            else:
                metrics["balance_over_lead_foot_norm"] = np.nan
        else:
            # Cannot normalize without address shoulder width
            metrics["balance_over_lead_foot_norm"] = np.nan
    else:
        metrics["balance_over_lead_foot_norm"] = np.nan

    # Shoulder finish angle: sh_line at finish; Angle(sh_line, target_vec)
    if l_shoulder is not None and r_shoulder is not None:
        sh_line = line_vec(r_shoulder, l_shoulder)
        if sh_line is not None:
            metrics["shoulder_finish_deg"] = acute_angle_deg(sh_line, target_vec)
        else:
            metrics["shoulder_finish_deg"] = np.nan
    else:
        metrics["shoulder_finish_deg"] = np.nan

    return metrics


def compute_metrics(
    swing_dict: Dict[str, np.ndarray], handedness: str = "right"
) -> Dict[str, Dict[str, float]]:
    """
    Compute swing metrics for all five positions.

    Args:
        swing_dict: Dictionary with keys "address", "top", "mid_ds", "impact", "finish"
                   Each value is a [1, 1, 17, 3] numpy array of keypoints. The target
                   vector is derived from the address ankles when available.
        handedness: "right" for right-handed (default), "left" for left-handed

    Returns:
        Nested dictionary with metrics per position:
        {
            "address": {"spine_forward_bend_deg": ..., "shoulder_alignment_deg": ...},
            "top": {"shoulder_turn_deg": ..., "pelvis_turn_deg": ...},
            "mid_ds": {"hip_open_deg": ..., "trail_elbow_flexion_deg": ...},
            "impact": {"hip_open_deg": ..., "forward_shaft_lean_deg": ...},
            "finish": {"balance_over_lead_foot_norm": ..., "shoulder_finish_deg": ...}
        }
    """
    result = {}

    # Get address and finish keypoints for baseline measurements
    address_keypoints = swing_dict.get("address")
    finish_keypoints = swing_dict.get("finish")
    target_vec = compute_target_vector(address_keypoints, handedness)

    # Compute metrics for each position
    if "address" in swing_dict:
        result["address"] = compute_address_metrics(
            swing_dict["address"], target_vec
        )

    if "top" in swing_dict and address_keypoints is not None:
        result["top"] = compute_top_metrics(
            swing_dict["top"], address_keypoints, handedness, finish_keypoints
        )
    elif "top" in swing_dict:
        result["top"] = {
            "shoulder_turn_deg": np.nan,
            "pelvis_turn_deg": np.nan,
        }

    if "mid_ds" in swing_dict:
        result["mid_ds"] = compute_mid_downswing_metrics(
            swing_dict["mid_ds"], target_vec, handedness
        )

    if "impact" in swing_dict:
        result["impact"] = compute_impact_metrics(
            swing_dict["impact"], target_vec, handedness
        )

    if "finish" in swing_dict and address_keypoints is not None:
        result["finish"] = compute_finish_metrics(
            swing_dict["finish"], address_keypoints, target_vec, handedness
        )
    elif "finish" in swing_dict:
        result["finish"] = {
            "balance_over_lead_foot_norm": np.nan,
            "shoulder_finish_deg": np.nan,
        }

    return result

