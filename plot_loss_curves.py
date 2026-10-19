"""
plot_loss_curves.py — Training Dynamics Visualisation.

Reads the CSV log produced by finetune.py and renders a 4-panel
publication-quality figure showing:
  Panel 1 (top-left)  — Total train vs val loss
  Panel 2 (top-right) — XYZ (spatial) loss
  Panel 3 (bottom-left) — Bone-length regularizer loss
  Panel 4 (bottom-right) — Joint Angle (accordion-fold) loss + MPJPE

Usage:
    python plot_loss_curves.py
    python plot_loss_curves.py --log results/training_log.csv --out results/loss_curves.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless-safe backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log(csv_path: Path) -> dict:
    """Read training_log.csv into a dict of float lists keyed by column name."""
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return data


def smooth(values: list, window: int = 3) -> np.ndarray:
    """Centred moving-average smoothing for noisy curves."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

PALETTE = {
    "train":  "#4A90D9",   # steel blue
    "val":    "#E85D75",   # coral red
    "xyz":    "#F5A623",   # amber
    "bone":   "#7ED321",   # lime
    "angle":  "#9B59B6",   # purple
    "mpjpe":  "#1ABC9C",   # teal
}


def style_ax(ax, title: str, ylabel: str):
    """Apply consistent dark-theme styling to a single axes."""
    ax.set_facecolor("#1A1A2E")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch", color="#AAAAAA", fontsize=9)
    ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=9)
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(True, color="#333355", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(
        frameon=True, facecolor="#2C2C3E", edgecolor="#555555",
        labelcolor="white", fontsize=8
    )


def plot(log: dict, out_path: Path):
    epochs = np.array(log["epoch"], dtype=int)

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8),
        facecolor="#0F0F1E",
        gridspec_kw={"hspace": 0.45, "wspace": 0.35}
    )
    fig.suptitle(
        "3D Hand Pose — Physics-Informed ViT Training Dynamics\n"
        "Joint Angle Regulariser: Accordion-Fold Suppression",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    # ── Panel 1: Total train / val loss ─────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, log["train_loss"], color=PALETTE["train"],
            linewidth=1.2, alpha=0.4, label="Train (raw)")
    ax.plot(epochs, smooth(log["train_loss"]), color=PALETTE["train"],
            linewidth=2.0, label="Train (smoothed)")
    ax.plot(epochs, log["val_loss"], color=PALETTE["val"],
            linewidth=1.2, alpha=0.4, label="Val (raw)")
    ax.plot(epochs, smooth(log["val_loss"]), color=PALETTE["val"],
            linewidth=2.0, label="Val (smoothed)")
    style_ax(ax, "Total Loss  (Train vs Val)", "Loss")

    # Mark best val epoch
    best_ep = int(np.argmin(log["val_loss"]))
    ax.axvline(epochs[best_ep], color="#FFFFFF", linewidth=0.8,
               linestyle=":", alpha=0.6)
    ax.annotate(
        f"Best\nEp {epochs[best_ep]}",
        xy=(epochs[best_ep], log["val_loss"][best_ep]),
        xytext=(epochs[best_ep] + max(1, len(epochs) // 12),
                log["val_loss"][best_ep]),
        color="white", fontsize=7,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.7),
    )

    # ── Panel 2: Per-component train loss breakdown ─────────────────────────
    ax = axes[0, 1]
    if "train_coord" in log:
        ax.plot(epochs, log["train_coord"], color=PALETTE["xyz"],
                linewidth=1.2, alpha=0.4, label="Coord (raw)")
        ax.plot(epochs, smooth(log["train_coord"]), color=PALETTE["xyz"],
                linewidth=2.0, label="Coord (smoothed)")
    style_ax(ax, "Coordinate Loss  (train)", "Loss")

    # ── Panel 3: Bone-length loss ────────────────────────────────────────────
    ax = axes[1, 0]
    if "train_bone" in log:
        ax.plot(epochs, log["train_bone"], color=PALETTE["bone"],
                linewidth=1.2, alpha=0.4, label="Bone (raw)")
        ax.plot(epochs, smooth(log["train_bone"]), color=PALETTE["bone"],
                linewidth=2.0, label="Bone (smoothed)")
    style_ax(ax, "Bone-Length Regularizer Loss", "Loss")

    # ── Panel 4: Angle loss + MPJPE (dual-axis) ──────────────────────────────
    ax = axes[1, 1]
    if "train_angle" in log:
        ax.plot(epochs, log["train_angle"], color=PALETTE["angle"],
                linewidth=1.2, alpha=0.4, label="Angle (raw)")
        ax.plot(epochs, smooth(log["train_angle"]), color=PALETTE["angle"],
                linewidth=2.0, label="Angle (smoothed)")
    style_ax(ax, "Joint Angle Loss  ·  Val MPJPE", "Angle Loss")

    # Second y-axis for MPJPE
    ax2 = ax.twinx()
    ax2.plot(epochs, log["val_mpjpe_mm"], color=PALETTE["mpjpe"],
             linewidth=1.8, linestyle="--", label="Val MPJPE (mm)")
    ax2.set_ylabel("MPJPE (mm)", color=PALETTE["mpjpe"], fontsize=9)
    ax2.tick_params(axis="y", colors=PALETTE["mpjpe"], labelsize=8)
    ax2.spines["right"].set_edgecolor(PALETTE["mpjpe"])

    # Merge legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        frameon=True, facecolor="#2C2C3E", edgecolor="#555555",
        labelcolor="white", fontsize=8
    )

    # Annotate final MPJPE
    final_mpjpe = log["val_mpjpe_mm"][-1]
    ax2.annotate(
        f"Final\n{final_mpjpe:.1f} mm",
        xy=(epochs[-1], final_mpjpe),
        xytext=(epochs[-1] - max(1, len(epochs) // 6), final_mpjpe * 1.08),
        color=PALETTE["mpjpe"], fontsize=7,
        arrowprops=dict(arrowstyle="->", color=PALETTE["mpjpe"], lw=0.7),
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    total_epochs = len(epochs)
    best_mpjpe   = min(log["val_mpjpe_mm"])
    best_val     = min(log["val_loss"])
    fig.text(
        0.5, 0.01,
        f"Total epochs: {total_epochs}  |  "
        f"Best val loss: {best_val:.4f}  |  "
        f"Best MPJPE: {best_mpjpe:.2f} mm  |  "
        f"λ_z=3  λ_bone=1.5  λ_angle=1.0",
        ha="center", color="#888888", fontsize=8
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PLOT] Saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument(
        "--log", type=str,
        default=str(config.RESULTS_DIR / "training_log.csv"),
        help="Path to training_log.csv produced by finetune.py"
    )
    parser.add_argument(
        "--out", type=str,
        default=str(config.RESULTS_DIR / "loss_curves.png"),
        help="Output PNG path"
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        print("        Run  python finetune.py --dataset <path>  first.")
        raise SystemExit(1)

    log = load_log(log_path)
    plot(log, out_path)


if __name__ == "__main__":
    main()
