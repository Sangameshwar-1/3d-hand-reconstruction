"""
post_train_pipeline.py — Automated Post-Training Pipeline.

Runs three steps in sequence after finetune.py completes:
  Step 1: Plot the 50-epoch loss curves  (plot_loss_curves.py logic)
  Step 2: Export best_model.pth -> hamer_champion.onnx  (export_onnx.py)
  Step 3: Re-run evaluation on the finetuned predictions (evaluate.py logic)

Usage (run manually OR chained after training):
    python post_train_pipeline.py --dataset dataset_aligned
    python post_train_pipeline.py --dataset dataset_aligned --skip-eval
"""

import argparse
import csv
import os
import subprocess
import sys
import json
import time
from pathlib import Path

import config

PROJECT = Path(__file__).resolve().parent

# Force UTF-8 I/O for all child processes so PyTorch's emoji prints
# (e.g. the \u2705 checkmark inside torch.onnx) don't crash on Windows cp1252.
UTF8_ENV = {**os.environ, "PYTHONUTF8": "1"}


def banner(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def step_plot():
    """Step 1: Generate the 4-panel loss curve PNG."""
    banner("Step 1/3 — Plotting Loss Curves")

    log_path = config.RESULTS_DIR / "training_log.csv"
    out_path = config.RESULTS_DIR / "loss_curves.png"

    if not log_path.exists():
        print(f"[SKIP] training_log.csv not found at {log_path}")
        print("       Make sure finetune.py completed successfully.")
        return False

    result = subprocess.run(
        [sys.executable, str(PROJECT / "plot_loss_curves.py"),
         "--log", str(log_path),
         "--out", str(out_path)],
        capture_output=False,
        env=UTF8_ENV,
    )

    if result.returncode != 0:
        print("[ERROR] plot_loss_curves.py failed.")
        return False

    print(f"[OK] Loss curves saved -> {out_path}")
    return True


def step_onnx_export():
    """Step 2: Export best_model.pth to hamer_champion.onnx."""
    banner("Step 2/3 — ONNX Export")

    weights = config.CHECKPOINTS_DIR / "best_model.pth"
    if not weights.exists():
        print(f"[ERROR] No checkpoint found at {weights}")
        print("        Ensure training completed and saved best_model.pth.")
        return False

    # Report checkpoint age / size for confidence
    stat = weights.stat()
    age_s = time.time() - stat.st_mtime
    print(f"[INFO] Checkpoint: {weights.name}  "
          f"size={stat.st_size/1e6:.1f} MB  "
          f"age={age_s/60:.1f} min")

    result = subprocess.run(
        [sys.executable, str(PROJECT / "export_onnx.py")],
        capture_output=False,
        env=UTF8_ENV,
    )

    if result.returncode != 0:
        print("[ERROR] export_onnx.py failed.")
        return False

    onnx_path = config.PROJECT_ROOT / "hamer_champion.onnx"
    if onnx_path.exists():
        print(f"[OK] ONNX model saved -> {onnx_path}  "
              f"({onnx_path.stat().st_size/1e6:.1f} MB)")
    return True


def step_evaluate(dataset_dir: Path):
    """Step 3: Run inference + evaluation on the finetuned model."""
    banner("Step 3/3 -- Final Evaluation (finetuned model)")

    # First generate finetuned predictions (inference.py --model finetuned)
    intrinsics = config.CALIBRATION_DIR / "camera_intrinsics.json"
    if not intrinsics.exists():
        print(f"[SKIP] Intrinsics not found at {intrinsics} -- skipping inference step.")
    else:
        print("[INFO] Running inference with finetuned model...")
        result = subprocess.run(
            [sys.executable, str(PROJECT / "inference.py"),
             "--dataset", str(dataset_dir),
             "--model", "finetuned",
             "--intrinsics", str(intrinsics)],
            capture_output=False,
            env=UTF8_ENV,
        )
        if result.returncode != 0:
            print("[ERROR] inference.py (finetuned) failed.")
            return False

    # Now run evaluation
    pred_dir = dataset_dir / "predictions_finetuned"
    if not pred_dir.exists():
        print(f"[SKIP] No predictions directory found at {pred_dir}")
        return False

    print("[INFO] Running MPJPE evaluation...")
    result = subprocess.run(
        [sys.executable, str(PROJECT / "evaluate.py"),
         "--dataset", str(dataset_dir),
         "--model", "finetuned"],
        capture_output=False,
        env=UTF8_ENV,
    )

    if result.returncode != 0:
        print("[ERROR] evaluate.py failed.")
        return False

    # Print the saved JSON results for a clean summary
    results_json = config.RESULTS_DIR / "eval_finetuned.json"
    if results_json.exists():
        with open(results_json) as f:
            r = json.load(f)
        banner("FINAL RESULTS SUMMARY")
        print(f"  Model           : {r['model']}")
        print(f"  Frames          : {r['frames_evaluated']}")
        print(f"  Overall MPJPE   : {r['overall_mpjpe_mm']:.2f} mm")
        print(f"  X-axis error    : {r['axis_errors_mm']['x']:.2f} mm")
        print(f"  Y-axis error    : {r['axis_errors_mm']['y']:.2f} mm")
        print(f"  Z-axis error    : {r['axis_errors_mm']['z']:.2f} mm  <- depth hallucination")
        print()
        print("  Per-joint breakdown:")
        for joint, err in r["per_joint_mpjpe_mm"].items():
            bar = "#" * int(err / 5)
            print(f"    {joint:<18s}  {err:6.2f} mm  {bar}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Post-training pipeline: plot -> ONNX export -> evaluate"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="dataset_aligned",
        help="Dataset directory (default: dataset_aligned)"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the inference + evaluation step"
    )
    args = parser.parse_args()

    dataset_dir = PROJECT / args.dataset

    print()
    print("=" * 60)
    print("  3D Hand Pose -- Post-Training Pipeline")
    print("  Steps: Loss Plot -> ONNX Export -> Evaluation")
    print("=" * 60)

    t0 = time.time()

    ok1 = step_plot()
    ok2 = step_onnx_export()
    ok3 = (not args.skip_eval) and step_evaluate(dataset_dir)

    elapsed = time.time() - t0
    banner(f"Pipeline Complete ({elapsed:.0f}s)")
    print(f"  Plot      : {'OK' if ok1 else 'FAILED / SKIPPED'}")
    print(f"  ONNX      : {'OK' if ok2 else 'FAILED'}")
    print(f"  Evaluate  : {'OK' if ok3 else 'SKIPPED' if args.skip_eval else 'FAILED'}")
    print()
    print("Next step -> python live_tracking.py")
    print("             Extend thumb horizontally and verify the 3D Isometric View.")


if __name__ == "__main__":
    main()
