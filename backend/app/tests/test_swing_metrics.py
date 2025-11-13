"""
Tests for swing metrics calculation module.
"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from app.processing.swing_metrics import (
    DEFAULT_TARGET_VEC,
    compute_metrics,
    compute_address_metrics,
    compute_top_metrics,
    compute_impact_metrics,
    compute_finish_metrics,
    compute_target_vector,
    extract_keypoint,
    angle_deg,
    line_angle_deg,
    KEYPOINT_THRESHOLD,
)
from app.processing.pose_estimation import estimate_pose


def load_keypoints_from_image(image_path: str) -> np.ndarray:
    """Load keypoints from an image using pose estimation."""
    img = Image.open(image_path)
    keypoints = estimate_pose(img)
    return keypoints


def test_extract_keypoint_valid():
    """Test extracting a valid keypoint."""
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    keypoints[0, 0, 5, :] = [0.5, 0.6, 0.8]  # y, x, score

    kpt = extract_keypoint(keypoints, 5)
    assert kpt is not None
    assert np.allclose(kpt, [0.6, 0.5])  # (x, y)


def test_extract_keypoint_low_confidence():
    """Test extracting keypoint with low confidence returns None."""
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    keypoints[0, 0, 5, :] = [0.5, 0.6, 0.05]  # score below threshold

    kpt = extract_keypoint(keypoints, 5)
    assert kpt is None


def test_extract_keypoint_missing():
    """Test extracting missing keypoint returns None."""
    keypoints = None
    kpt = extract_keypoint(keypoints, 5)
    assert kpt is None


def test_angle_deg():
    """Test angle calculation between vectors."""
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    angle = angle_deg(vec1, vec2)
    assert np.isclose(angle, 90.0, atol=0.1)

    # Parallel vectors
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([2.0, 0.0])
    angle = angle_deg(vec1, vec2)
    assert np.isclose(angle, 0.0, atol=0.1)


def test_angle_deg_none():
    """Test angle calculation with None vectors."""
    angle = angle_deg(None, np.array([1.0, 0.0]))
    assert np.isnan(angle)


def test_line_angle_deg():
    """Test line angle calculation."""
    # Horizontal line (aligned with target vector (1, 0))
    pt1 = np.array([0.0, 0.5])  # Left point
    pt2 = np.array([1.0, 0.5])  # Right point (horizontal)
    angle = line_angle_deg(pt1, pt2, DEFAULT_TARGET_VEC)
    assert np.isclose(angle, 0.0, atol=1.0)


def test_compute_target_vector_from_ankles_right_handed():
    """Target vector should align with ankle line for right-handed golfer."""
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Lead (left) ankle and trail (right) ankle define horizontal line
    address_keypoints[0, 0, 15, :] = [0.7, 0.4, 0.9]  # left ankle (lead)
    address_keypoints[0, 0, 16, :] = [0.7, 0.8, 0.9]  # right ankle (trail)

    target_vec = compute_target_vector(address_keypoints, handedness="right")

    assert np.allclose(target_vec, np.array([1.0, 0.0]), atol=1e-6)


def test_compute_target_vector_left_handed_orientation():
    """Left-handed golfers should flip lead/trail ankle mapping."""
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Lead ankle is right ankle for left-handed golfers
    address_keypoints[0, 0, 15, :] = [0.7, 0.2, 0.9]  # left ankle (trail)
    address_keypoints[0, 0, 16, :] = [0.7, 0.6, 0.9]  # right ankle (lead)

    target_vec = compute_target_vector(address_keypoints, handedness="left")

    assert np.allclose(target_vec, np.array([1.0, 0.0]), atol=1e-6)


def test_compute_target_vector_fallback_without_ankles():
    """Missing ankle data should return default target vector."""
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    target_vec = compute_target_vector(address_keypoints)

    assert np.allclose(target_vec, DEFAULT_TARGET_VEC)


def test_impact_metrics_respect_dynamic_target_line():
    """Hip rotation should be measured relative to the derived target vector."""
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Create ankles forming a 30° upward-sloping target line
    address_keypoints[0, 0, 15, :] = [0.7, 0.6, 0.9]  # left ankle (lead)
    address_keypoints[0, 0, 16, :] = [0.8, 0.3, 0.9]  # right ankle (trail)

    target_vec = compute_target_vector(address_keypoints)

    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Align hips along target vector so hip_open_deg ≈ 0
    right_hip = np.array([0.6, 0.4])
    hip_offset = target_vec * 0.2
    left_hip = right_hip + hip_offset
    keypoints[0, 0, 11, :] = [left_hip[1], left_hip[0], 0.9]  # left hip (y, x, score)
    keypoints[0, 0, 12, :] = [right_hip[1], right_hip[0], 0.9]  # right hip (y, x, score)

    # Provide lead arm keypoints to keep other metrics valid
    keypoints[0, 0, 7, :] = [0.5, 0.5, 0.9]
    keypoints[0, 0, 9, :] = [0.4, 0.5, 0.9]

    metrics = compute_impact_metrics(keypoints, target_vec)

    assert "hip_open_deg" in metrics
    assert np.isfinite(metrics["hip_open_deg"])
    assert abs(metrics["hip_open_deg"]) <= 5.0, metrics["hip_open_deg"]


def test_compute_address_metrics_synthetic_square():
    """Test address metrics with synthetic square pose."""
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # Create a square pose: shoulders and hips horizontal, aligned
    # Nose at (0.5, 0.3)
    keypoints[0, 0, 0, :] = [0.3, 0.5, 0.9]  # nose

    # Shoulders horizontal at y=0.4
    keypoints[0, 0, 5, :] = [0.4, 0.4, 0.9]  # left shoulder
    keypoints[0, 0, 6, :] = [0.4, 0.6, 0.9]  # right shoulder

    # Hips horizontal at y=0.6
    keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip

    metrics = compute_address_metrics(keypoints, DEFAULT_TARGET_VEC)

    # Shoulder line should be horizontal, so angle to target line should be ~0°
    assert "shoulder_alignment_deg" in metrics
    # For horizontal line, angle to horizontal target line should be close to 0°
    assert np.isclose(metrics["shoulder_alignment_deg"], 0.0, atol=1.0)

    # Spine forward bend should be reasonable (forward lean)
    assert "spine_forward_bend_deg" in metrics
    assert not np.isnan(metrics["spine_forward_bend_deg"])


def test_compute_top_metrics_ankle_referenced():
    """Test top metrics with ankle-referenced rotation geometry."""
    # Address position: shoulders and hips horizontal, aligned with ankle line
    # Ankles form horizontal target line (left→right)
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 15, :] = [0.7, 0.3, 0.9]  # left ankle (y, x, score)
    address_keypoints[0, 0, 16, :] = [0.7, 0.7, 0.9]  # right ankle
    address_keypoints[0, 0, 5, :] = [0.4, 0.3, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.7, 0.9]  # right shoulder
    address_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    address_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip

    # Top position: shoulders rotated ~90° counterclockwise from ankle line
    # Ankle line is horizontal (0.3→0.7), so 90° rotation means vertical
    # Center at (0.5, 0.4), vertical line means same x, different y
    top_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    top_keypoints[0, 0, 5, :] = [0.3, 0.5, 0.9]  # left shoulder (vertical, above center)
    top_keypoints[0, 0, 6, :] = [0.5, 0.5, 0.9]  # right shoulder (vertical, below center)
    
    # Top position: hips rotated ~45° counterclockwise from ankle line
    # For 45° rotation from horizontal: equal x and y components
    # Center at (0.5, 0.6), distance from center = 0.1
    # Vector: (0.1 * cos(45°), 0.1 * sin(45°)) = (0.0707, 0.0707)
    top_keypoints[0, 0, 11, :] = [0.6 - 0.0707, 0.5 - 0.0707, 0.9]  # left hip
    top_keypoints[0, 0, 12, :] = [0.6 + 0.0707, 0.5 + 0.0707, 0.9]  # right hip

    metrics = compute_top_metrics(top_keypoints, address_keypoints, handedness="right")

    # Shoulder turn should be ~90° (rotated from horizontal ankle line to vertical)
    assert "shoulder_turn_deg" in metrics
    assert not np.isnan(metrics["shoulder_turn_deg"])
    assert 85.0 <= abs(metrics["shoulder_turn_deg"]) <= 95.0, f"Expected ~90°, got {metrics['shoulder_turn_deg']:.2f}°"

    # Pelvis turn should be ~45°
    assert "pelvis_turn_deg" in metrics
    assert not np.isnan(metrics["pelvis_turn_deg"])
    assert 40.0 <= abs(metrics["pelvis_turn_deg"]) <= 50.0, f"Expected ~45°, got {metrics['pelvis_turn_deg']:.2f}°"


def test_compute_top_metrics_with_ankle_fallback():
    """Test that address ankles are always used as baseline, even if top ankles are missing."""
    # Address position: horizontal alignment
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 15, :] = [0.7, 0.3, 0.9]  # left ankle (y, x, score)
    address_keypoints[0, 0, 16, :] = [0.7, 0.7, 0.9]  # right ankle
    address_keypoints[0, 0, 5, :] = [0.4, 0.3, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.7, 0.9]  # right shoulder
    address_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    address_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip

    # Top position: shoulders rotated ~60° counterclockwise
    # Ankle line is horizontal, so 60° rotation means rotated line
    # Center at (0.5, 0.4), for 60°: cos(60°)=0.5, sin(60°)=0.866
    # Vector from center: (0.2 * 0.5, 0.2 * 0.866) = (0.1, 0.173)
    top_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    top_keypoints[0, 0, 5, :] = [0.4 - 0.173, 0.5 - 0.1, 0.9]  # left shoulder
    top_keypoints[0, 0, 6, :] = [0.4 + 0.173, 0.5 + 0.1, 0.9]  # right shoulder
    
    # Top position: hips rotated ~30° counterclockwise
    # Center at (0.5, 0.6), for 30°: cos(30°)=0.866, sin(30°)=0.5
    # Vector from center: (0.1 * 0.866, 0.1 * 0.5) = (0.0866, 0.05)
    top_keypoints[0, 0, 11, :] = [0.6 - 0.05, 0.5 - 0.0866, 0.9]  # left hip
    top_keypoints[0, 0, 12, :] = [0.6 + 0.05, 0.5 + 0.0866, 0.9]  # right hip
    # Note: top ankles are not provided - should use address ankles

    metrics = compute_top_metrics(top_keypoints, address_keypoints, handedness="right")

    # Shoulder turn should be ~60°
    assert "shoulder_turn_deg" in metrics
    assert not np.isnan(metrics["shoulder_turn_deg"])
    assert 55.0 <= abs(metrics["shoulder_turn_deg"]) <= 65.0, f"Expected ~60°, got {metrics['shoulder_turn_deg']:.2f}°"

    # Pelvis turn should be ~30°
    assert "pelvis_turn_deg" in metrics
    assert not np.isnan(metrics["pelvis_turn_deg"])
    assert 25.0 <= abs(metrics["pelvis_turn_deg"]) <= 35.0, f"Expected ~30°, got {metrics['pelvis_turn_deg']:.2f}°"


def test_compute_impact_metrics_known_angles():
    """Test impact metrics with known rotation angles."""
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # Create hips rotated ~40° from horizontal target line
    # For 40° from horizontal: tan(40°) ≈ 0.839
    # If horizontal distance is 0.2, vertical offset is 0.2 * 0.839 ≈ 0.168
    # Line from right hip to left hip: going left (dx = -0.2)
    # For 40° rotation, dy should be such that the angle is 40°
    # In image coords (y-down), if we want 40° from horizontal going left:
    # The line vector should be (-0.2 * cos(40°), 0.2 * sin(40°)) = (-0.153, 0.129)
    # So if right hip is at (0.6, 0.6), left hip should be at (0.6 - 0.153, 0.6 + 0.129) = (0.447, 0.729)
    # But let's simplify: use horizontal distance 0.2, vertical offset 0.168
    # Right hip at (0.6, 0.6), left hip at (0.4, 0.6 + 0.168) = (0.4, 0.768)
    keypoints[0, 0, 11, :] = [0.768, 0.4, 0.9]  # left hip (y, x, score)
    keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip (y, x, score)

    # Shoulders rotated ~15° from horizontal target line
    # tan(15°) ≈ 0.268, if horizontal distance is 0.1, vertical offset is 0.1 * 0.268 ≈ 0.027
    # Right shoulder at (0.4, 0.55), left shoulder at (0.45, 0.4 + 0.027) = (0.45, 0.427)
    keypoints[0, 0, 5, :] = [0.427, 0.45, 0.9]  # left shoulder (y, x, score)
    keypoints[0, 0, 6, :] = [0.4, 0.55, 0.9]  # right shoulder (y, x, score)

    # Lead forearm (elbow to wrist) - vertical (going up in image, decreasing y)
    keypoints[0, 0, 7, :] = [0.5, 0.5, 0.9]  # left elbow (y, x, score)
    keypoints[0, 0, 9, :] = [0.4, 0.5, 0.9]  # left wrist (vertical, going up, y decreases)

    metrics = compute_impact_metrics(keypoints, DEFAULT_TARGET_VEC)

    assert "hip_open_deg" in metrics
    assert not np.isnan(metrics["hip_open_deg"])
    # Should be approximately 40° ± 5° (may vary slightly due to coordinate system)
    assert 35.0 <= metrics["hip_open_deg"] <= 45.0

    assert "forward_shaft_lean_deg" in metrics
    assert not np.isnan(metrics["forward_shaft_lean_deg"])
    # Should be close to 0° (vertical)
    assert metrics["forward_shaft_lean_deg"] <= 5.0


def test_compute_finish_metrics_elbow_angle():
    """Test finish metrics with straight lead arm."""
    # Address position: shoulder width = 0.4 (from x=0.3 to x=0.7)
    # In normalized coords, this is 0.4. For test, we'll use this as baseline.
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 5, :] = [0.4, 0.3, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.7, 0.9]  # right shoulder

    finish_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # Straight lead arm: shoulder, elbow, wrist in line
    finish_keypoints[0, 0, 5, :] = [0.3, 0.5, 0.9]  # left shoulder
    finish_keypoints[0, 0, 7, :] = [0.4, 0.5, 0.9]  # left elbow (straight line)
    finish_keypoints[0, 0, 9, :] = [0.5, 0.5, 0.9]  # left wrist (straight line)

    finish_keypoints[0, 0, 6, :] = [0.3, 0.6, 0.9]  # right shoulder
    # Hip center directly over lead ankle: both at x=0.5
    finish_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    finish_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    # Hip center is at x = (0.4 + 0.6) / 2 = 0.5
    finish_keypoints[0, 0, 15, :] = [0.7, 0.5, 0.9]  # left ankle (same x as hip center)

    metrics = compute_finish_metrics(
        finish_keypoints, address_keypoints, DEFAULT_TARGET_VEC
    )

    # Balance over lead foot should be ~0 (hip center x - ankle x = 0.5 - 0.5 = 0)
    assert "balance_over_lead_foot_norm" in metrics
    assert not np.isnan(metrics["balance_over_lead_foot_norm"])
    # Normalized by shoulder width (0.4), so 0 / 0.4 = 0
    assert np.isclose(metrics["balance_over_lead_foot_norm"], 0.0, atol=0.1)

    # Shoulder finish angle should be computed
    assert "shoulder_finish_deg" in metrics
    assert not np.isnan(metrics["shoulder_finish_deg"])


def test_compute_metrics_missing_keypoints():
    """Test that missing keypoints return np.nan."""
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Set all scores below threshold
    keypoints[0, 0, :, 2] = 0.05

    swing_dict = {
        "address": keypoints,
        "top": keypoints,
        "impact": keypoints,
        "finish": keypoints,
    }

    result = compute_metrics(swing_dict)

    # All metrics should be np.nan
    for position in ["address", "top", "impact", "finish"]:
        assert position in result
        for metric_value in result[position].values():
            assert np.isnan(metric_value)


def test_compute_metrics_real_images():
    """Test metrics computation with real test images."""
    test_dir = Path(__file__).parent

    # Check if test images exist
    address_img = test_dir / "video_1346_event_0_address.png"
    top_img = test_dir / "video_1346_event_3_top.png"
    impact_img = test_dir / "video_1346_event_5_impact.png"
    finish_img = test_dir / "video_1346_event_7_finish.png"

    if not all(
        [
            address_img.exists(),
            top_img.exists(),
            impact_img.exists(),
            finish_img.exists(),
        ]
    ):
        pytest.skip("Test images not found")

    # Load keypoints from images
    address_keypoints = load_keypoints_from_image(str(address_img))
    top_keypoints = load_keypoints_from_image(str(top_img))
    impact_keypoints = load_keypoints_from_image(str(impact_img))
    finish_keypoints = load_keypoints_from_image(str(finish_img))

    swing_dict = {
        "address": address_keypoints,
        "top": top_keypoints,
        "impact": impact_keypoints,
        "finish": finish_keypoints,
    }

    result = compute_metrics(swing_dict)

    # Check that all positions are present
    assert "address" in result
    assert "top" in result
    assert "impact" in result
    assert "finish" in result

    # Check address metrics
    assert "spine_forward_bend_deg" in result["address"]
    assert "shoulder_alignment_deg" in result["address"]

    # Check top metrics
    assert "shoulder_turn_deg" in result["top"]
    assert "pelvis_turn_deg" in result["top"]

    # Check impact metrics
    assert "hip_open_deg" in result["impact"]
    assert "forward_shaft_lean_deg" in result["impact"]

    # Check finish metrics
    assert "balance_over_lead_foot_norm" in result["finish"]
    assert "shoulder_finish_deg" in result["finish"]

    # Verify that metrics are finite numbers (not all NaN)
    # At least some metrics should be valid
    has_valid_metrics = False
    for position_metrics in result.values():
        for value in position_metrics.values():
            if not np.isnan(value) and np.isfinite(value):
                has_valid_metrics = True
                break
        if has_valid_metrics:
            break

    assert has_valid_metrics, "At least some metrics should be valid"


def test_compute_metrics_missing_address():
    """Test that metrics can be computed even if address is missing."""
    # Create valid keypoints for impact (doesn't need address)
    impact_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    impact_keypoints[0, 0, :, 2] = 0.9  # High confidence
    # Set valid coordinates for impact metrics
    impact_keypoints[0, 0, 5, :] = [0.4, 0.4, 0.9]  # left shoulder
    impact_keypoints[0, 0, 6, :] = [0.4, 0.6, 0.9]  # right shoulder
    impact_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    impact_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    impact_keypoints[0, 0, 7, :] = [0.5, 0.5, 0.9]  # left elbow
    impact_keypoints[0, 0, 9, :] = [0.55, 0.5, 0.9]  # left wrist

    # Create keypoints for top and finish (will have NaN without address)
    top_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    top_keypoints[0, 0, :, 2] = 0.9
    finish_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    finish_keypoints[0, 0, :, 2] = 0.9

    swing_dict = {
        "top": top_keypoints,
        "impact": impact_keypoints,
        "finish": finish_keypoints,
    }

    result = compute_metrics(swing_dict)

    # Top and finish should have NaN metrics (need address for baselines)
    assert "top" in result
    assert np.isnan(result["top"]["shoulder_turn_deg"])

    # Impact should still work (doesn't need address)
    assert "impact" in result
    assert not all(np.isnan(v) for v in result["impact"].values())


def test_compute_impact_metrics_left_handed():
    """Test impact metrics for left-handed golfer (uses right arm)."""
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # Create hips and shoulders (symmetric, same for both)
    keypoints[0, 0, 11, :] = [0.768, 0.4, 0.9]  # left hip
    keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    keypoints[0, 0, 5, :] = [0.427, 0.45, 0.9]  # left shoulder
    keypoints[0, 0, 6, :] = [0.4, 0.55, 0.9]  # right shoulder

    # For left-handed: lead arm is RIGHT arm (R_ELBOW -> R_WRIST)
    # Lead forearm (right arm) - vertical (going up in image, decreasing y)
    keypoints[0, 0, 8, :] = [0.5, 0.5, 0.9]  # right elbow (y, x, score)
    keypoints[0, 0, 10, :] = [0.4, 0.5, 0.9]  # right wrist (vertical, going up)

    # Test with left-handed parameter
    metrics = compute_impact_metrics(keypoints, DEFAULT_TARGET_VEC, handedness="left")

    assert "hip_open_deg" in metrics
    assert not np.isnan(metrics["hip_open_deg"])

    assert "forward_shaft_lean_deg" in metrics
    assert not np.isnan(metrics["forward_shaft_lean_deg"])
    # Should be close to 0° (vertical) for left-handed using right arm
    assert metrics["forward_shaft_lean_deg"] <= 5.0

    # Compare with right-handed (should use left arm)
    keypoints[0, 0, 7, :] = [0.5, 0.5, 0.9]  # left elbow
    keypoints[0, 0, 9, :] = [0.4, 0.5, 0.9]  # left wrist
    metrics_right = compute_impact_metrics(keypoints, DEFAULT_TARGET_VEC, handedness="right")
    
    # Both should have similar forward_shaft_lean_deg since both arms are vertical
    assert abs(metrics["forward_shaft_lean_deg"] - metrics_right["forward_shaft_lean_deg"]) < 1.0


def test_compute_finish_metrics_left_handed():
    """Test finish metrics for left-handed golfer (uses right arm and ankle)."""
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 5, :] = [0.4, 0.3, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.7, 0.9]  # right shoulder

    finish_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # For left-handed: lead arm is RIGHT arm
    # Straight lead arm: right shoulder, right elbow, right wrist in line
    finish_keypoints[0, 0, 6, :] = [0.3, 0.5, 0.9]  # right shoulder
    finish_keypoints[0, 0, 8, :] = [0.4, 0.5, 0.9]  # right elbow (straight line)
    finish_keypoints[0, 0, 10, :] = [0.5, 0.5, 0.9]  # right wrist (straight line)

    finish_keypoints[0, 0, 5, :] = [0.3, 0.4, 0.9]  # left shoulder
    # Hip center directly over lead ankle (right ankle for left-handed): both at x=0.5
    finish_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    finish_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    # Hip center is at x = (0.4 + 0.6) / 2 = 0.5
    finish_keypoints[0, 0, 16, :] = [0.7, 0.5, 0.9]  # right ankle (same x as hip center)

    metrics = compute_finish_metrics(
        finish_keypoints, address_keypoints, DEFAULT_TARGET_VEC, handedness="left"
    )

    # Balance over lead foot should be ~0 (hip center x - right ankle x = 0.5 - 0.5 = 0)
    assert "balance_over_lead_foot_norm" in metrics
    assert not np.isnan(metrics["balance_over_lead_foot_norm"])
    # Normalized by shoulder width (0.4), so 0 / 0.4 = 0
    assert np.isclose(metrics["balance_over_lead_foot_norm"], 0.0, atol=0.1)

    # Shoulder finish angle should be computed
    assert "shoulder_finish_deg" in metrics
    assert not np.isnan(metrics["shoulder_finish_deg"])


def test_compute_metrics_left_handed():
    """Test full metrics computation for left-handed golfer."""
    # Create keypoints for all positions
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 0, :] = [0.3, 0.5, 0.9]  # nose
    address_keypoints[0, 0, 5, :] = [0.4, 0.4, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.6, 0.9]  # right shoulder
    address_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    address_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    # Ankles for ankle-referenced rotation calculation (left-handed: right ankle is lead)
    address_keypoints[0, 0, 15, :] = [0.7, 0.4, 0.9]  # left ankle (trail)
    address_keypoints[0, 0, 16, :] = [0.7, 0.6, 0.9]  # right ankle (lead)

    top_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    top_keypoints[0, 0, 5, :] = [0.4, 0.4, 0.9]  # left shoulder
    top_keypoints[0, 0, 6, :] = [0.4, 0.6, 0.9]  # right shoulder
    top_keypoints[0, 0, 11, :] = [0.6, 0.415, 0.9]  # left hip
    top_keypoints[0, 0, 12, :] = [0.6, 0.585, 0.9]  # right hip

    impact_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    impact_keypoints[0, 0, 5, :] = [0.427, 0.45, 0.9]  # left shoulder
    impact_keypoints[0, 0, 6, :] = [0.4, 0.55, 0.9]  # right shoulder
    impact_keypoints[0, 0, 11, :] = [0.768, 0.4, 0.9]  # left hip
    impact_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    # For left-handed: lead arm is RIGHT arm
    impact_keypoints[0, 0, 8, :] = [0.5, 0.5, 0.9]  # right elbow
    impact_keypoints[0, 0, 10, :] = [0.4, 0.5, 0.9]  # right wrist

    finish_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    finish_keypoints[0, 0, 5, :] = [0.3, 0.4, 0.9]  # left shoulder
    finish_keypoints[0, 0, 6, :] = [0.3, 0.6, 0.9]  # right shoulder
    finish_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    finish_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    # For left-handed: lead arm and ankle are RIGHT side
    finish_keypoints[0, 0, 6, :] = [0.3, 0.5, 0.9]  # right shoulder
    finish_keypoints[0, 0, 8, :] = [0.4, 0.5, 0.9]  # right elbow
    finish_keypoints[0, 0, 10, :] = [0.5, 0.5, 0.9]  # right wrist
    finish_keypoints[0, 0, 16, :] = [0.7, 0.5, 0.9]  # right ankle

    swing_dict = {
        "address": address_keypoints,
        "top": top_keypoints,
        "impact": impact_keypoints,
        "finish": finish_keypoints,
    }

    # Test with left-handed parameter
    result = compute_metrics(swing_dict, handedness="left")

    # Check that all positions are present
    assert "address" in result
    assert "top" in result
    assert "impact" in result
    assert "finish" in result

    # Check that metrics are computed (not all NaN)
    assert not all(np.isnan(v) for v in result["address"].values())
    assert not all(np.isnan(v) for v in result["top"].values())
    assert not all(np.isnan(v) for v in result["impact"].values())
    assert not all(np.isnan(v) for v in result["finish"].values())

    # Verify left-handed specific metrics use right arm/ankle
    # Forward shaft lean should be computed from right arm
    assert "forward_shaft_lean_deg" in result["impact"]
    assert not np.isnan(result["impact"]["forward_shaft_lean_deg"])

    # Balance over lead foot should use right ankle
    assert "balance_over_lead_foot_norm" in result["finish"]
    assert not np.isnan(result["finish"]["balance_over_lead_foot_norm"])


def test_compute_metrics_handedness_comparison():
    """Test that metrics differ appropriately between right and left-handed golfers."""
    # Create keypoints where left and right arms are in different positions
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 5, :] = [0.4, 0.3, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.7, 0.9]  # right shoulder

    impact_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # Left arm at 45° angle
    impact_keypoints[0, 0, 7, :] = [0.5, 0.4, 0.9]  # left elbow
    impact_keypoints[0, 0, 9, :] = [0.45, 0.35, 0.9]  # left wrist (at angle)
    # Right arm vertical
    impact_keypoints[0, 0, 8, :] = [0.5, 0.6, 0.9]  # right elbow
    impact_keypoints[0, 0, 10, :] = [0.4, 0.6, 0.9]  # right wrist (vertical)

    finish_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    finish_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    finish_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    # Left ankle at x=0.3, right ankle at x=0.7
    finish_keypoints[0, 0, 15, :] = [0.7, 0.3, 0.9]  # left ankle
    finish_keypoints[0, 0, 16, :] = [0.7, 0.7, 0.9]  # right ankle

    swing_dict = {
        "address": address_keypoints,
        "impact": impact_keypoints,
        "finish": finish_keypoints,
    }

    # Test right-handed (uses left arm/ankle)
    result_right = compute_metrics(swing_dict, handedness="right")
    
    # Test left-handed (uses right arm/ankle)
    result_left = compute_metrics(swing_dict, handedness="left")

    # Forward shaft lean should differ (left arm at 45° vs right arm vertical)
    assert result_right["impact"]["forward_shaft_lean_deg"] > 10.0  # Left arm at angle
    assert result_left["impact"]["forward_shaft_lean_deg"] < 5.0  # Right arm vertical

    # Balance over lead foot should differ (different ankles)
    # Hip center at x=0.5, left ankle at x=0.3, right ankle at x=0.7
    # Right-handed: (0.5 - 0.3) / 0.4 = 0.5
    # Left-handed: (0.5 - 0.7) / 0.4 = -0.5
    assert result_right["finish"]["balance_over_lead_foot_norm"] > 0
    assert result_left["finish"]["balance_over_lead_foot_norm"] < 0

