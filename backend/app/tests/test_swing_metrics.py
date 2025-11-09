"""
Tests for swing metrics calculation module.
"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from app.processing.swing_metrics import (
    compute_metrics,
    compute_address_metrics,
    compute_top_metrics,
    compute_impact_metrics,
    compute_finish_metrics,
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
    # Vertical line going up (aligned with target vector (0, -1))
    # In image coords (y-down), going up means decreasing y
    pt1 = np.array([0.0, 1.0])  # Lower point
    pt2 = np.array([0.0, 0.0])  # Upper point (going up)
    angle = line_angle_deg(pt1, pt2)
    assert np.isclose(abs(angle), 0.0, atol=1.0)


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

    metrics = compute_address_metrics(keypoints)

    # Shoulder line should be horizontal, so angle to vertical should be ~90°
    assert "shoulder_to_target_deg" in metrics
    # For horizontal line, angle to vertical should be close to 90°
    assert not np.isnan(metrics["shoulder_to_target_deg"])

    # Spine tilt should be reasonable (forward lean)
    assert "spine_tilt_deg" in metrics
    assert not np.isnan(metrics["spine_tilt_deg"])

    # Pelvis center x should be normalized
    assert "pelvis_center_x_norm" in metrics
    assert not np.isnan(metrics["pelvis_center_x_norm"])


def test_compute_top_metrics_foreshortening():
    """Test top metrics with foreshortening (shoulder width halved)."""
    # Address position: full shoulder width = 0.4 (from x=0.3 to x=0.7)
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 5, :] = [0.4, 0.3, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.7, 0.9]  # right shoulder
    address_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    address_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip

    # Top position: shoulder width halved = 0.2 (from x=0.4 to x=0.6)
    # This gives ratio = 0.2/0.4 = 0.5, so arccos(0.5) ≈ 60°
    top_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    top_keypoints[0, 0, 5, :] = [0.4, 0.4, 0.9]  # left shoulder (moved toward center)
    top_keypoints[0, 0, 6, :] = [0.4, 0.6, 0.9]  # right shoulder (moved toward center)
    top_keypoints[0, 0, 11, :] = [0.6, 0.45, 0.9]  # left hip
    top_keypoints[0, 0, 12, :] = [0.6, 0.55, 0.9]  # right hip

    metrics = compute_top_metrics(top_keypoints, address_keypoints)

    # Shoulder turn should be ~60° (arccos(0.5) ≈ 60°)
    assert "shoulder_turn_deg" in metrics
    assert not np.isnan(metrics["shoulder_turn_deg"])
    # Should be approximately 60 degrees
    assert 55.0 <= metrics["shoulder_turn_deg"] <= 65.0

    # X-factor should be computed
    assert "x_factor_deg" in metrics
    assert not np.isnan(metrics["x_factor_deg"])


def test_compute_impact_metrics_known_angles():
    """Test impact metrics with known rotation angles."""
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # Create hips rotated ~40° from vertical
    # For 40° rotation, if vertical is (0, -1), rotated line is approximately
    # at angle. We'll set points to create this angle.
    # Hip line rotated 40°: points at angle
    keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip

    # Shoulders rotated ~15° from vertical
    keypoints[0, 0, 5, :] = [0.4, 0.45, 0.9]  # left shoulder
    keypoints[0, 0, 6, :] = [0.4, 0.55, 0.9]  # right shoulder

    # Lead forearm (elbow to wrist)
    keypoints[0, 0, 7, :] = [0.5, 0.5, 0.9]  # left elbow
    keypoints[0, 0, 9, :] = [0.55, 0.5, 0.9]  # left wrist (vertical)

    metrics = compute_impact_metrics(keypoints)

    assert "hip_open_deg" in metrics
    assert not np.isnan(metrics["hip_open_deg"])

    assert "shoulder_open_deg" in metrics
    assert not np.isnan(metrics["shoulder_open_deg"])

    assert "forward_lean_deg" in metrics
    assert not np.isnan(metrics["forward_lean_deg"])


def test_compute_finish_metrics_elbow_angle():
    """Test finish metrics with straight lead arm."""
    address_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    address_keypoints[0, 0, 5, :] = [0.4, 0.3, 0.9]  # left shoulder
    address_keypoints[0, 0, 6, :] = [0.4, 0.7, 0.9]  # right shoulder

    finish_keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # Straight lead arm: shoulder, elbow, wrist in line
    finish_keypoints[0, 0, 5, :] = [0.3, 0.5, 0.9]  # left shoulder
    finish_keypoints[0, 0, 7, :] = [0.4, 0.5, 0.9]  # left elbow (straight line)
    finish_keypoints[0, 0, 9, :] = [0.5, 0.5, 0.9]  # left wrist (straight line)

    finish_keypoints[0, 0, 6, :] = [0.3, 0.6, 0.9]  # right shoulder
    finish_keypoints[0, 0, 11, :] = [0.6, 0.4, 0.9]  # left hip
    finish_keypoints[0, 0, 12, :] = [0.6, 0.6, 0.9]  # right hip
    finish_keypoints[0, 0, 15, :] = [0.7, 0.4, 0.9]  # left ankle

    metrics = compute_finish_metrics(finish_keypoints, address_keypoints)

    # Elbow angle should be close to 180° (straight arm)
    assert "lead_elbow_angle_deg" in metrics
    assert not np.isnan(metrics["lead_elbow_angle_deg"])
    assert metrics["lead_elbow_angle_deg"] >= 170.0


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
    assert "spine_tilt_deg" in result["address"]
    assert "shoulder_to_target_deg" in result["address"]
    assert "pelvis_center_x_norm" in result["address"]

    # Check top metrics
    assert "shoulder_turn_deg" in result["top"]
    assert "pelvis_turn_deg" in result["top"]
    assert "x_factor_deg" in result["top"]

    # Check impact metrics
    assert "hip_open_deg" in result["impact"]
    assert "shoulder_open_deg" in result["impact"]
    assert "forward_lean_deg" in result["impact"]

    # Check finish metrics
    assert "balance_offset_norm" in result["finish"]
    assert "shoulder_finish_deg" in result["finish"]
    assert "lead_elbow_angle_deg" in result["finish"]

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

