"""
fist_metrics.py — Rigorous 3D Hand Pose Evaluation Metrics.

Implements:
  1. MPJPE (Mean Per-Joint Position Error) in millimetres.
  2. PA-MPJPE (Procrustes-Aligned MPJPE) in millimetres via closed-form
     Orthogonal Procrustes analysis (Umeyama algorithm).
  3. Per-Axis Error (X, Y, Z Mean Absolute Error).
  4. Per-Finger Error breakdown (Thumb, Index, Middle, Ring, Pinky).
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch

import config


def compute_mpjpe(
    gt: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Mean Per-Joint Position Error (MPJPE).

    Args:
        gt: Ground-truth joints array (21, 3) or (B, 21, 3) in millimetres.
        pred: Predicted joints array (21, 3) or (B, 21, 3) in millimetres.

    Returns:
        mean_mpjpe (float): Overall MPJPE in mm.
        per_joint_err (np.ndarray): (21,) array of mean error per joint in mm.
        per_axis_err (np.ndarray): (3,) array of MAE [X, Y, Z] in mm.
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    gt = gt.reshape(-1, config.NUM_JOINTS, 3)
    pred = pred.reshape(-1, config.NUM_JOINTS, 3)

    # Euclidean distance per joint: (B, 21)
    diff = gt - pred
    dist = np.linalg.norm(diff, axis=-1)

    mean_mpjpe = float(np.mean(dist))
    per_joint_err = np.mean(dist, axis=0)  # (21,)
    per_axis_err = np.mean(np.abs(diff), axis=(0, 1))  # (3,) [X, Y, Z]

    return mean_mpjpe, per_joint_err, per_axis_err


def compute_procrustes_alignment(
    source: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    Compute optimal similarity transformation (rigid rotation R, translation t,
    and scale s) that aligns source onto target using the Umeyama algorithm.

    argmin_{s, R, t} || target - (s * source @ R.T + t) ||_F^2

    Args:
        source: (N, 3) candidate joint array to align.
        target: (N, 3) reference ground-truth joint array.

    Returns:
        aligned_source: (N, 3) transformed points aligned to target.
    """
    # 1. Centroid subtraction
    mu_s = np.mean(source, axis=0, keepdims=True)
    mu_t = np.mean(target, axis=0, keepdims=True)

    s_centered = source - mu_s
    t_centered = target - mu_t

    # 2. Covariance matrix
    H = s_centered.T @ t_centered  # (3, 3)

    # 3. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection check (ensure det(R) == +1); the flipped axis also flips the
    # sign of its singular value in the optimal scale (Umeyama 1991, eq. 42)
    signs = np.ones(3)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        signs[2] = -1.0

    # 4. Optimal scale
    var_s = np.sum(s_centered ** 2)
    scale = np.sum(S * signs) / (var_s + 1e-8)

    # 5. Optimal translation
    t = mu_t - scale * (mu_s @ R.T)

    aligned = scale * (source @ R.T) + t
    return aligned


def compute_pa_mpjpe(
    gt: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, np.ndarray]:
    """
    Compute Procrustes-Aligned Mean Per-Joint Position Error (PA-MPJPE).
    Removes global rotation, scale, and translation discrepancy to isolate
    pure articulation / pose reconstruction accuracy.

    Args:
        gt: Ground-truth joints (B, 21, 3) or (21, 3) in mm.
        pred: Predicted joints (B, 21, 3) or (21, 3) in mm.

    Returns:
        mean_pa_mpjpe (float): Overall PA-MPJPE in mm.
        per_joint_pa (np.ndarray): (21,) array of mean PA error per joint in mm.
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    gt = gt.reshape(-1, config.NUM_JOINTS, 3)
    pred = pred.reshape(-1, config.NUM_JOINTS, 3)

    B = gt.shape[0]
    aligned_preds = np.zeros_like(pred)

    for b in range(B):
        aligned_preds[b] = compute_procrustes_alignment(source=pred[b], target=gt[b])

    dist = np.linalg.norm(gt - aligned_preds, axis=-1)  # (B, 21)
    mean_pa_mpjpe = float(np.mean(dist))
    per_joint_pa = np.mean(dist, axis=0)

    return mean_pa_mpjpe, per_joint_pa


def format_evaluation_summary(
    gt: np.ndarray,
    pred: np.ndarray
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Generate a publication-ready dictionary of all error metrics."""
    mpjpe, pj_err, pa_err = compute_mpjpe(gt, pred)
    pa_mpjpe, pj_pa = compute_pa_mpjpe(gt, pred)

    finger_errors = {}
    for finger_name, (start_idx, end_idx) in config.FINGER_RANGES.items():
        finger_errors[finger_name] = round(float(np.mean(pj_err[start_idx:end_idx + 1])), 2)

    return {
        "mpjpe_mm": round(mpjpe, 2),
        "pa_mpjpe_mm": round(pa_mpjpe, 2),
        "axis_errors_mm": {
            "x": round(float(pa_err[0]), 2),
            "y": round(float(pa_err[1]), 2),
            "z_depth": round(float(pa_err[2]), 2),
        },
        "per_finger_mpjpe_mm": finger_errors,
        "per_joint_mpjpe_mm": {
            config.JOINT_NAMES[i]: round(float(pj_err[i]), 2)
            for i in range(config.NUM_JOINTS)
        },
    }
