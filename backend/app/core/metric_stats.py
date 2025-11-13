"""
Statistics module for computing z-scores from swing metrics.
Contains hard-coded mean and standard deviation values for each metric.
"""
from typing import Dict, Optional


# Hard-coded statistics for each metric
# Format: {metric_name: {"mean": float, "std": float}}
# Initialize all values to 0.0 - user will fill in later
METRIC_STATISTICS: Dict[str, Dict[str, float]] = {
    "address_spine_forward_bend_deg": {"mean": 0.0, "std": 0.0},
    "address_shoulder_alignment_deg": {"mean": 0.0, "std": 0.0},
    "top_shoulder_turn_deg": {"mean": 0.0, "std": 0.0},
    "top_pelvis_turn_deg": {"mean": 0.0, "std": 0.0},
    "mid_ds_hip_open_deg": {"mean": 0.0, "std": 0.0},
    "mid_ds_trail_elbow_flexion_deg": {"mean": 0.0, "std": 0.0},
    "impact_hip_open_deg": {"mean": 0.0, "std": 0.0},
    "impact_forward_shaft_lean_deg": {"mean": 0.0, "std": 0.0},
    "finish_balance_over_lead_foot_norm": {"mean": 0.0, "std": 0.0},
}


def compute_z_score(metric_name: str, value: Optional[float]) -> float:
    """
    Compute z-score for a metric value.
    
    Args:
        metric_name: Name of the metric (e.g., "address_spine_forward_bend_deg")
        value: The metric value (can be None for missing metrics)
    
    Returns:
        z-score: (value - mean) / std, or 0.0 if value is None, std is 0, or metric not found
    """
    if value is None:
        return 0.0
    
    stats = METRIC_STATISTICS.get(metric_name)
    if not stats:
        return 0.0
    
    mean = stats["mean"]
    std = stats["std"]
    
    # If std is 0, return 0.0 to avoid division by zero
    if std == 0.0:
        return 0.0
    
    return (value - mean) / std


def get_metric_statistics(metric_name: str) -> Optional[Dict[str, float]]:
    """
    Get statistics for a specific metric.
    
    Args:
        metric_name: Name of the metric
    
    Returns:
        Dictionary with "mean" and "std" keys, or None if metric not found
    """
    return METRIC_STATISTICS.get(metric_name)

