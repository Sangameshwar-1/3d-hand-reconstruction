"""
eval_fist.py — Comprehensive Scientific Evaluation Benchmark for 3DFist.

Evaluates 3DFist predictions against Ultraleap 3D ground-truth coordinates.
Outputs:
  1. Overall MPJPE (Mean Per-Joint Position Error) in mm.
  2. PA-MPJPE (Procrustes-Aligned MPJPE) in mm.
  3. Orthogonal Axis Error Decomposition (X width, Y height, Z depth).
  4. Per-Finger and Per-Joint error distributions under self-occlusion.
  5. JSON summary (results/eval_3dfist.json).
  6. Publication-ready comparison plot (results/eval_plot_3dfist.png).

Usage:
  python eval_fist.py --dataset dataset_ir --weights checkpoints/fist_best.pth
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from fist_model import load_fist_checkpoint
from fist_dataset import load_split_frames
from hand_locator import HandLocator
from ir_transforms import StereoPerspectivePreprocessor
from fist_metrics import (
    compute_mpjpe,
    compute_pa_mpjpe,
    format_evaluation_summary,
)


def plot_fist_results(
    pj_err: np.ndarray,
    pj_pa_err: np.ndarray,
    pa_axis_err: np.ndarray,
    out_path: Path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0e1117")
    fig.suptitle("3DFist — Dual-IR 3D Hand Pose Benchmark", color="white", fontsize=13, fontweight="bold")

    # Dark background styling
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, color="#21262d", linestyle="--", alpha=0.7)

    # 1. Per-Joint Error (MPJPE vs PA-MPJPE)
    x = np.arange(config.NUM_JOINTS)
    width = 0.38
    labels = [config.JOINT_NAMES[i].replace("_", "\n") for i in range(config.NUM_JOINTS)]

    ax1.bar(x - width/2, pj_err, width, label="MPJPE (raw)", color="#58a6ff", edgecolor="none")
    ax1.bar(x + width/2, pj_pa_err, width, label="PA-MPJPE (rigid-aligned)", color="#3fb950", edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)
    ax1.set_ylabel("Error (mm)", color="#c9d1d9", fontsize=9)
    ax1.set_title("Per-Joint Error Distribution (Self-Occlusion Analysis)", color="white", fontsize=10, fontweight="bold")
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    # 2. X vs Y vs Z Component Error
    ax2.bar(["X (Width)", "Y (Height)", "Z (Depth)"], pa_axis_err, color=["#f0883e", "#d2a8ff", "#238636"], width=0.5)
    ax2.set_ylabel("Mean Absolute Error (mm)", color="#c9d1d9", fontsize=9)
    ax2.set_title("Orthogonal Axis Error Breakdown", color="white", fontsize=10, fontweight="bold")
    for i, v in enumerate(pa_axis_err):
        ax2.text(i, v + 0.3, f"{v:.2f} mm", ha="center", color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"[EVAL] Benchmark plot saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="3DFist Scientific Evaluation Benchmark")
    parser.add_argument("--dataset", type=str, default="dataset_ir",
                        help="Dataset directory")
    parser.add_argument("--weights", type=str, default="checkpoints/fist_best.pth",
                        help="Path to trained weights file")
    parser.add_argument("--out", type=str, default="results/eval_3dfist.json",
                        help="Output JSON summary path")
    parser.add_argument("--handloc", type=str, default=None,
                        help="HandLocNet weights: crop with the IR-only locator instead of the "
                             "projected Ultraleap joints (oracle crop)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found at {weights_path}")
        print("        Train a model first with: python train_fist.py")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"[EVAL] Initializing 3DFist model on {device}...")

    try:
        model, _ = load_fist_checkpoint(weights_path, device)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Held-out split (same session split as train_fist.py). Predictions are made in
    # the virtual camera frame and rotated back, so all errors are in LeapC axes.
    all_preds = []
    all_gts = []
    missed = 0
    preprocessor = StereoPerspectivePreprocessor(target_size=config.IR_INPUT_SIZE, is_training=False)
    frames = load_split_frames(args.dataset, "val", 0.2)
    locator = HandLocator(args.handloc, device) if args.handloc else None
    crop_mode = "handlocnet" if locator else "oracle_projected_joints"
    print(f"[EVAL] Benchmarking on {len(frames)} frames ({crop_mode} perspective crops)...")
    with torch.no_grad():
        for img_l, img_r, joints, bbox_l, bbox_r, luts in frames:
            if locator is not None:
                bbox_l, bbox_r, _ = locator.locate_pair(img_l, img_r)
            out = None
            if bbox_l is not None and bbox_r is not None:
                out = preprocessor(img_l, img_r, luts, bbox_l, bbox_r)
            if out is None:
                missed += 1
                continue
            x, geo, geom = out
            pred_virtual = model(x[None], geo[None])[0].numpy()
            all_preds.append((pred_virtual @ geom.leap_to_virtual)[None])
            all_gts.append((joints - joints[0:1])[None])

    if not all_preds:
        print("[ERROR] No frames could be evaluated (no hand found / boxes not triangulable).")
        sys.exit(1)
    preds = np.concatenate(all_preds, axis=0)  # (N, 21, 3)
    gts = np.concatenate(all_gts, axis=0)      # (N, 21, 3)

    # Reference: always predicting the mean training pose. A model that does not
    # beat this has learned nothing from the images.
    train_frames = load_split_frames(args.dataset, "train", 0.2)
    mean_pose = np.mean([f[2] - f[2][0:1] for f in train_frames], axis=0)
    baseline_mpjpe = float(np.linalg.norm(gts - mean_pose[None], axis=-1).mean())

    # Compute comprehensive metrics
    mpjpe, pj_err, pa_axis_err = compute_mpjpe(gts, preds)
    pa_mpjpe, pj_pa_err = compute_pa_mpjpe(gts, preds)

    summary = format_evaluation_summary(gts, preds)
    summary["evaluated_samples"] = len(preds)
    summary["missed_by_locator"] = missed
    summary["crop_mode"] = crop_mode
    summary["mean_pose_baseline_mpjpe_mm"] = round(baseline_mpjpe, 2)
    summary["model_parameters"] = sum(p.numel() for p in model.parameters())

    # Print Report
    print()
    print("=" * 64)
    print("  3DFist — SCIENTIFIC EVALUATION REPORT (PALMAR STEREO IR)")
    print("=" * 64)
    print(f"  Samples Evaluated   : {len(preds)}")
    print(f"  Model Parameters    : {summary['model_parameters']:,}")
    print("-" * 64)
    print(f"  Crop                : {crop_mode}" + (f" ({missed} frames missed)" if missed else ""))
    print(f"  Overall MPJPE       : {mpjpe:6.2f} mm")
    print(f"  Mean-pose baseline  : {baseline_mpjpe:6.2f} mm  (must be beaten)")
    print(f"  PA-MPJPE            : {pa_mpjpe:6.2f} mm  (Rigid-Aligned)")
    print("-" * 64)
    print("  Per-Axis Error (MAE):")
    print(f"    X (Lateral Width) : {pa_axis_err[0]:6.2f} mm")
    print(f"    Y (Height)        : {pa_axis_err[1]:6.2f} mm")
    print(f"    Z (Optical Depth) : {pa_axis_err[2]:6.2f} mm  <-- Stereo Triangulation")
    print("-" * 64)
    print("  Per-Finger Breakdown (MPJPE):")
    for finger, err in summary["per_finger_mpjpe_mm"].items():
        bar = "#" * int(err / 2)
        print(f"    {finger:<10s} : {err:5.2f} mm  {bar}")
    print("=" * 64)

    # Save JSON summary
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[EVAL] Metrics saved -> {out_file}")

    # Generate visual figure
    plot_path = out_file.parent / out_file.name.replace("eval_", "eval_plot_").replace(".json", ".png")
    plot_fist_results(pj_err, pj_pa_err, pa_axis_err, plot_path)


if __name__ == "__main__":
    main()
