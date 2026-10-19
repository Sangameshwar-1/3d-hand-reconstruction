"""
evaluate.py — Stage 4b: MPJPE Error Evaluation.

Calculates the Mean Per Joint Position Error (MPJPE) between the ground truth
(Leap) and the predicted 3D joints. Both are root-aligned to the wrist (Joint 0)
before calculation to remove global translation differences.

Usage:
    python evaluate.py --dataset dataset/ --model mediapipe
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset_utils import load_dataset_pairs, load_joint_json, joints_dict_to_array


def align_to_root(joints: np.ndarray, root_idx: int = config.WRIST_JOINT_INDEX) -> np.ndarray:
    """
    Root align a (21, 3) joint array by subtracting the root joint position.
    """
    root_pos = np.copy(joints[root_idx])
    return joints - root_pos


def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> tuple:
    """
    Compute MPJPE and per-joint errors.
    
    Formula:
      MPJPE = (1/21) * Sum(sqrt((x_gt - x_pred)^2 + (y_gt - y_pred)^2 + (z_gt - z_pred)^2))
      
    Args:
        gt: (21, 3) ground truth array (mm)
        pred: (21, 3) prediction array (mm)
        
    Returns:
        (mpjpe, per_joint_errors, per_axis_errors)
    """
    # Euclidean distance per joint: shape (21,)
    per_joint_err = np.linalg.norm(gt - pred, axis=1)
    
    # MPJPE scalar
    mpjpe = float(np.mean(per_joint_err))
    
    # Per-axis error (absolute difference): shape (21, 3)
    per_axis_err = np.abs(gt - pred)
    
    return mpjpe, per_joint_err, per_axis_err


def evaluate_dataset(dataset_dir: Path, model_name: str):
    """Run evaluation over all frames with predictions."""
    pairs = load_dataset_pairs(dataset_dir)
    pred_dir = dataset_dir / f"predictions_{model_name}"
    
    if not pred_dir.exists():
        print(f"[ERROR] Predictions directory not found: {pred_dir}")
        sys.exit(1)
        
    frame_mpjpes = []
    all_per_joint_errs = []
    all_per_axis_errs = []
    
    print(f"\n[EVALUATE] Computing MPJPE for {model_name}...")
    
    for rgb_path, leap_path in pairs:
        pred_path = pred_dir / (rgb_path.stem + ".json")
        
        if not pred_path.exists():
            continue
            
        # Load GT
        gt_data = load_joint_json(leap_path)
        gt_joints = joints_dict_to_array(gt_data["joints"])
        
        # Load Pred
        pred_data = load_joint_json(pred_path)
        pred_joints = joints_dict_to_array(pred_data["joints"])
        
        # Root Align
        gt_aligned = align_to_root(gt_joints)
        pred_aligned = align_to_root(pred_joints)
        
        # Compute Error
        mpjpe, pj_err, pa_err = compute_mpjpe(gt_aligned, pred_aligned)
        
        frame_mpjpes.append(mpjpe)
        all_per_joint_errs.append(pj_err)
        all_per_axis_errs.append(pa_err)
        
    if not frame_mpjpes:
        print("[ERROR] No overlapping ground truth and predictions found.")
        return
        
    # Aggregate stats
    mean_mpjpe = np.mean(frame_mpjpes)
    mean_pj_err = np.mean(all_per_joint_errs, axis=0)  # (21,)
    mean_pa_err = np.mean(all_per_axis_errs, axis=0)   # (21, 3)
    
    print("\n" + "="*50)
    print(f"  Evaluation Results: {model_name}")
    print("="*50)
    print(f"  Frames Evaluated: {len(frame_mpjpes)}")
    print(f"  Overall MPJPE:    {mean_mpjpe:.2f} mm\n")
    
    print("  Per-Axis Error Breakdown (Mean Abs Error):")
    avg_x = np.mean(mean_pa_err[:, 0])
    avg_y = np.mean(mean_pa_err[:, 1])
    avg_z = np.mean(mean_pa_err[:, 2])
    print(f"    X-axis (Width):  {avg_x:.2f} mm")
    print(f"    Y-axis (Height): {avg_y:.2f} mm")
    print(f"    Z-axis (Depth):  {avg_z:.2f} mm  <-- Z Hallucination")
    
    # Save results to JSON
    results = {
        "model": model_name,
        "frames_evaluated": len(frame_mpjpes),
        "overall_mpjpe_mm": round(float(mean_mpjpe), 2),
        "axis_errors_mm": {
            "x": round(float(avg_x), 2),
            "y": round(float(avg_y), 2),
            "z": round(float(avg_z), 2),
        },
        "per_joint_mpjpe_mm": {
            config.JOINT_NAMES[i]: round(float(mean_pj_err[i]), 2)
            for i in range(config.NUM_JOINTS)
        }
    }
    
    out_file = config.RESULTS_DIR / f"eval_{model_name}.json"
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVALUATE] Results saved to {out_file}")
    
    # Plotting
    plot_results(mean_pj_err, mean_pa_err, model_name)


def plot_results(mean_pj_err: np.ndarray, mean_pa_err: np.ndarray, model_name: str):
    """Generate Matplotlib charts for the evaluation results."""
    # Ensure fonts are readable
    plt.rcParams.update({'font.size': 10})
    
    fig = plt.figure(figsize=(15, 6))
    
    # 1. Per-Joint Error Bar Chart
    ax1 = plt.subplot(1, 2, 1)
    x_pos = np.arange(config.NUM_JOINTS)
    labels = [config.JOINT_NAMES[i].replace("_", "\n") for i in range(config.NUM_JOINTS)]
    
    ax1.bar(x_pos, mean_pj_err, color='skyblue', edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)
    ax1.set_ylabel("MPJPE (mm)")
    ax1.set_title(f"Per-Joint Error - {model_name}")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. X, Y, Z Component Error
    ax2 = plt.subplot(1, 2, 2)
    width = 0.25
    
    ax2.bar(x_pos - width, mean_pa_err[:, 0], width, label='X Error', color='tomato')
    ax2.bar(x_pos,         mean_pa_err[:, 1], width, label='Y Error', color='lightgreen')
    ax2.bar(x_pos + width, mean_pa_err[:, 2], width, label='Z Error (Depth)', color='royalblue')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)
    ax2.set_ylabel("Absolute Error (mm)")
    ax2.set_title("X vs Y vs Z Error Analysis")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = config.RESULTS_DIR / f"eval_plot_{model_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[EVALUATE] Plots saved to {plot_path}")
    # plt.show()  # Uncomment to view interactively


def main():
    parser = argparse.ArgumentParser(description="Stage 4b: MPJPE Error Evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--model", type=str, default="mediapipe",
                        help="Baseline model to evaluate (e.g. mediapipe, hamer)")
    args = parser.parse_args()

    evaluate_dataset(Path(args.dataset), args.model)


if __name__ == "__main__":
    main()
