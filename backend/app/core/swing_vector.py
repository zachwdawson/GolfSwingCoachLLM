"""
Swing vector builder module.
Creates a 16-dimensional vector from swing metrics for similarity matching.
"""
from typing import Dict, Any, List, Optional
from app.core.metric_stats import compute_z_score

# Order of dimensions in the metrics_vector (must match build_vector_db.py)
VECTOR_FIELDS = [
    "contact_normal",
    "contact_fat",
    "contact_thin",
    "contact_inconsistent",
    "ball_shape_normal",
    "ball_shape_hook",
    "ball_shape_slice",
    "address_spine_forward_bend_deg",
    "address_shoulder_alignment_deg",
    "top_shoulder_turn_deg",
    "top_pelvis_turn_deg",
    "mid_ds_hip_open_deg",
    "mid_ds_trail_elbow_flexion_deg",
    "impact_hip_open_deg",
    "impact_forward_shaft_lean_deg",
    "finish_balance_over_lead_foot_norm",
]

# Indices (for convenience)
IDX = {name: i for i, name in enumerate(VECTOR_FIELDS)}

# Contact and ball-shape categories
CONTACT_CATS = ["normal", "fat", "thin", "inconsistent"]
BALL_SHAPE_CATS = ["normal", "hook", "slice"]


def build_swing_vector(
    metrics_by_position: Dict[str, Dict[str, Any]],
    contact: str = "normal",
    ball_shape: str = "normal"
) -> List[float]:
    """
    Build a 16-dimensional vector from swing metrics.
    
    Args:
        metrics_by_position: Dictionary with keys "address", "top", "mid_ds", "impact", "finish"
                            Each value is a dict of metric_name -> value
        contact: Contact type (default: "normal")
        ball_shape: Ball shape (default: "normal")
    
    Returns:
        16-dimensional vector matching the swing_patterns structure
    """
    vec = [0.0] * len(VECTOR_FIELDS)
    
    # 1) One-hot encode contact (default to "normal")
    contact_lower = contact.lower() if contact else "normal"
    if contact_lower in CONTACT_CATS:
        vec[IDX[f"contact_{contact_lower}"]] = 1.0
    else:
        vec[IDX["contact_normal"]] = 1.0
    
    # 2) One-hot encode ball shape (default to "normal")
    ball_shape_lower = ball_shape.lower() if ball_shape else "normal"
    if ball_shape_lower in BALL_SHAPE_CATS:
        vec[IDX[f"ball_shape_{ball_shape_lower}"]] = 1.0
    else:
        vec[IDX["ball_shape_normal"]] = 1.0
    
    # 3) Extract metrics from each position and compute z-scores
    # Address metrics
    address_metrics = metrics_by_position.get("address", {})
    if address_metrics:
        # address_spine_forward_bend_deg
        spine_bend = address_metrics.get("spine_forward_bend_deg")
        if spine_bend is not None:
            vec[IDX["address_spine_forward_bend_deg"]] = compute_z_score(
                "address_spine_forward_bend_deg", spine_bend
            )
        
        # address_shoulder_alignment_deg
        shoulder_align = address_metrics.get("shoulder_alignment_deg")
        if shoulder_align is not None:
            vec[IDX["address_shoulder_alignment_deg"]] = compute_z_score(
                "address_shoulder_alignment_deg", shoulder_align
            )
    
    # Top metrics
    top_metrics = metrics_by_position.get("top", {})
    if top_metrics:
        # top_shoulder_turn_deg
        shoulder_turn = top_metrics.get("shoulder_turn_deg")
        if shoulder_turn is not None:
            vec[IDX["top_shoulder_turn_deg"]] = compute_z_score(
                "top_shoulder_turn_deg", shoulder_turn
            )
        
        # top_pelvis_turn_deg
        pelvis_turn = top_metrics.get("pelvis_turn_deg")
        if pelvis_turn is not None:
            vec[IDX["top_pelvis_turn_deg"]] = compute_z_score(
                "top_pelvis_turn_deg", pelvis_turn
            )
    
    # Mid-downswing metrics
    mid_ds_metrics = metrics_by_position.get("mid_ds", {})
    if mid_ds_metrics:
        # mid_ds_hip_open_deg
        hip_open = mid_ds_metrics.get("hip_open_deg")
        if hip_open is not None:
            vec[IDX["mid_ds_hip_open_deg"]] = compute_z_score(
                "mid_ds_hip_open_deg", hip_open
            )
        
        # mid_ds_trail_elbow_flexion_deg
        elbow_flex = mid_ds_metrics.get("trail_elbow_flexion_deg")
        if elbow_flex is not None:
            vec[IDX["mid_ds_trail_elbow_flexion_deg"]] = compute_z_score(
                "mid_ds_trail_elbow_flexion_deg", elbow_flex
            )
    
    # Impact metrics
    impact_metrics = metrics_by_position.get("impact", {})
    if impact_metrics:
        # impact_hip_open_deg
        impact_hip_open = impact_metrics.get("hip_open_deg")
        if impact_hip_open is not None:
            vec[IDX["impact_hip_open_deg"]] = compute_z_score(
                "impact_hip_open_deg", impact_hip_open
            )
        
        # impact_forward_shaft_lean_deg
        shaft_lean = impact_metrics.get("forward_shaft_lean_deg")
        if shaft_lean is not None:
            vec[IDX["impact_forward_shaft_lean_deg"]] = compute_z_score(
                "impact_forward_shaft_lean_deg", shaft_lean
            )
    
    # Finish metrics
    finish_metrics = metrics_by_position.get("finish", {})
    if finish_metrics:
        # finish_balance_over_lead_foot_norm
        balance = finish_metrics.get("balance_over_lead_foot_norm")
        if balance is not None:
            vec[IDX["finish_balance_over_lead_foot_norm"]] = compute_z_score(
                "finish_balance_over_lead_foot_norm", balance
            )
    
    return vec

