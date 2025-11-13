"""
Process CSV file with keypoints and generate metric distributions.

This script:
1. Loads keypoints from driver_dtl_with_keypoints.csv
2. Computes swing metrics for each record
3. Generates distribution plots (histograms) for each metric
4. Saves plots as PNG files
5. Saves distribution statistics in result.csv
"""
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PNG generation
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from app.processing.swing_metrics import compute_metrics, compute_target_vector


def parse_keypoints_from_csv_row(pose_keypoints_str: str) -> Dict[str, np.ndarray]:
    """
    Parse keypoints from CSV row (JSON string format).
    
    Args:
        pose_keypoints_str: JSON string containing array of frames with keypoints
        
    Returns:
        Dictionary mapping position labels to keypoint arrays [1, 1, 17, 3]
    """
    try:
        frames = json.loads(pose_keypoints_str)
    except (json.JSONDecodeError, TypeError):
        return {}
    
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


def process_csv_file(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Process CSV file and compute metrics for each row.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of dictionaries containing row data and computed metrics
    """
    results = []
    
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} rows")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx + 1}/{len(df)}...")
        
        # Parse keypoints from pose_keypoints column
        pose_keypoints_str = row.get("pose_keypoints", "")
        if pd.isna(pose_keypoints_str) or not pose_keypoints_str:
            continue
        
        swing_dict = parse_keypoints_from_csv_row(pose_keypoints_str)
        
        # Skip if we don't have the required positions
        if "address" not in swing_dict:
            continue
        
        # Compute metrics
        try:
            metrics_result = compute_metrics(swing_dict, handedness="right")
            
            # Store results with row metadata
            result_entry = {
                "id": row.get("id", idx),
                "youtube_id": row.get("youtube_id", ""),
                "player": row.get("player", ""),
                "sex": row.get("sex", ""),
                "club": row.get("club", ""),
                "view": row.get("view", ""),
            }
            
            # Flatten metrics into result entry
            for position, metrics in metrics_result.items():
                for metric_name, metric_value in metrics.items():
                    key = f"{position}_{metric_name}"
                    result_entry[key] = metric_value if not np.isnan(metric_value) else None
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"Successfully processed {len(results)} rows")
    return results


def collect_metrics_for_distribution(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Collect all metric values across all results for distribution analysis.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary mapping metric names to lists of values
    """
    metrics_dict = defaultdict(list)
    
    for result in results:
        for key, value in result.items():
            # Skip metadata columns
            if key in ["id", "youtube_id", "player", "sex", "club", "view"]:
                continue
            
            # Only collect numeric values (not None)
            if value is not None and isinstance(value, (int, float)):
                metrics_dict[key].append(float(value))
    
    return dict(metrics_dict)


def generate_distribution_plots(metrics_dict: Dict[str, List[float]], output_dir: Path):
    """
    Generate histogram plots for each metric and save as PNG files.
    
    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
        output_dir: Directory to save PNG files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating distribution plots...")
    
    for metric_name, values in metrics_dict.items():
        if len(values) == 0:
            continue
        
        # Filter out any remaining NaN or infinite values
        clean_values = [v for v in values if np.isfinite(v)]
        
        if len(clean_values) == 0:
            continue
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(clean_values, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel(metric_name.replace("_", " ").title())
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {metric_name.replace('_', ' ').title()}\n(n={len(clean_values)})")
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(clean_values)
        std_val = np.std(clean_values)
        median_val = np.median(clean_values)
        stats_text = f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        safe_filename = metric_name.replace("/", "_").replace("\\", "_")
        plot_path = output_dir / f"{safe_filename}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if len(metrics_dict) <= 20 or metric_name in list(metrics_dict.keys())[:5]:
            print(f"  Saved: {plot_path.name}")


def compute_distribution_statistics(metrics_dict: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Compute distribution statistics for each metric.
    
    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
        
    Returns:
        DataFrame with statistics for each metric
    """
    stats_rows = []
    
    for metric_name, values in metrics_dict.items():
        if len(values) == 0:
            continue
        
        # Filter out any remaining NaN or infinite values
        clean_values = [v for v in values if np.isfinite(v)]
        
        if len(clean_values) == 0:
            continue
        
        stats = {
            "metric": metric_name,
            "count": len(clean_values),
            "mean": np.mean(clean_values),
            "std": np.std(clean_values),
            "median": np.median(clean_values),
            "min": np.min(clean_values),
            "max": np.max(clean_values),
            "q25": np.percentile(clean_values, 25),
            "q75": np.percentile(clean_values, 75),
        }
        stats_rows.append(stats)
    
    return pd.DataFrame(stats_rows)


def main():
    """Main function to process CSV and generate distributions."""
    # Get paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "driver_dtl_with_keypoints.csv"
    output_dir = script_dir
    result_csv_path = script_dir / "result.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Process CSV file
    print("=" * 60)
    print("PROCESSING CSV FILE")
    print("=" * 60)
    results = process_csv_file(csv_path)
    
    if len(results) == 0:
        print("Error: No valid results found")
        return
    
    # Collect metrics for distribution
    print("\n" + "=" * 60)
    print("COLLECTING METRICS")
    print("=" * 60)
    metrics_dict = collect_metrics_for_distribution(results)
    print(f"Found {len(metrics_dict)} unique metrics")
    
    # Generate distribution plots
    print("\n" + "=" * 60)
    print("GENERATING DISTRIBUTION PLOTS")
    print("=" * 60)
    generate_distribution_plots(metrics_dict, output_dir)
    
    # Compute and save statistics
    print("\n" + "=" * 60)
    print("COMPUTING DISTRIBUTION STATISTICS")
    print("=" * 60)
    stats_df = compute_distribution_statistics(metrics_dict)
    stats_df.to_csv(result_csv_path, index=False)
    print(f"Saved statistics to: {result_csv_path}")
    print(f"\nSummary:")
    print(f"  Total metrics: {len(stats_df)}")
    print(f"  Total records processed: {len(results)}")
    
    # Print sample statistics
    print("\nSample statistics (first 10 metrics):")
    print(stats_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

