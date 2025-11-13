"""
Test script to generate metrics from test_points_1392.json and print them
in the same format as the frontend page.tsx.
"""
import json
import numpy as np
from pathlib import Path
from app.processing.swing_metrics import compute_metrics


def load_keypoints_from_json(json_path: str) -> dict:
    """Load frames and keypoints from JSON file."""
    with open(json_path, 'r') as f:
        frames = json.load(f)
    
    # Convert keypoints to numpy arrays in the format expected by compute_metrics
    # Keypoints are [y, x, score] for each of 17 keypoints
    # Need to reshape to [1, 1, 17, 3]
    result = {}
    
    # Map JSON labels to position keys used by compute_metrics
    label_to_position = {
        "address": "address",
        "top": "top",
        "mid_ds": "mid_ds",
        "impact": "impact",
        "finish": "finish",
    }
    
    for frame in frames:
        label = frame.get("label", "").lower()
        if label in label_to_position:
            keypoints_list = frame.get("keypoints", [])
            if len(keypoints_list) == 17:
                # Convert to numpy array: [1, 1, 17, 3] with [y, x, score] format
                keypoints_array = np.array(keypoints_list, dtype=np.float32)
                # Reshape to [1, 1, 17, 3]
                keypoints_reshaped = keypoints_array.reshape(1, 1, 17, 3)
                result[label_to_position[label]] = keypoints_reshaped
    
    return result


def format_metrics(metrics_result: dict) -> str:
    """
    Format metrics in the same way as the frontend page.tsx.
    
    Args:
        metrics_result: Dictionary with position keys and their metrics
        
    Returns:
        Formatted string with metrics
    """
    # Map position keys to display names (matching frontend)
    position_display_map = {
        "address": "Address",
        "top": "Top",
        "mid_ds": "Mid-downswing",
        "impact": "Impact",
        "finish": "Finish",
    }
    
    # Order for display (matching frontend)
    position_order = ["address", "top", "mid_ds", "impact", "finish"]
    
    output = ""
    for position_key in position_order:
        if position_key in metrics_result:
            display_name = position_display_map.get(position_key, position_key)
            output += f"{display_name}:\n"
            
            metrics = metrics_result[position_key]
            for key, value in sorted(metrics.items()):
                if value is not None and not np.isnan(value):
                    # Format metric name (remove _deg suffix for display, add ° symbol)
                    display_key = key.replace("_deg", "").replace("_", " ")
                    if key.endswith("_deg") or "angle" in key.lower():
                        display_value = f"{value:.1f}°"
                    elif isinstance(value, (int, float)):
                        display_value = f"{value:.2f}"
                    else:
                        display_value = str(value)
                    output += f"  - {display_key}: {display_value}\n"
            output += "\n"
    
    return output or "No metrics available"


def main():
    """Main function to load JSON, compute metrics, and print results."""
    # Get the path to the test JSON file
    test_dir = Path(__file__).parent
    json_path = test_dir / "test_points_1392.json"
    
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return
    
    print(f"Loading keypoints from {json_path}...")
    swing_dict = load_keypoints_from_json(str(json_path))
    
    if not swing_dict:
        print("Error: No valid frames found in JSON file")
        return
    
    print(f"Found {len(swing_dict)} positions: {', '.join(swing_dict.keys())}\n")
    
    # Compute metrics
    print("Computing swing metrics...")
    metrics_result = compute_metrics(swing_dict, handedness="right")
    
    # Format and print metrics
    print("\n" + "=" * 60)
    print("SWING METRICS")
    print("=" * 60 + "\n")
    formatted_output = format_metrics(metrics_result)
    print(formatted_output)
    
    # Also print raw metrics for debugging
    print("\n" + "=" * 60)
    print("RAW METRICS (for debugging)")
    print("=" * 60 + "\n")
    for position, metrics in metrics_result.items():
        print(f"{position}:")
        for key, value in sorted(metrics.items()):
            if np.isnan(value):
                print(f"  {key}: NaN")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

