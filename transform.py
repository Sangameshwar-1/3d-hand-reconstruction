"""
transform.py — Stage 3c: Extrinsic Transformation Utilities.

Applies the extrinsic calibration matrix to transform LeapC 3D coordinates
(origin = Leap device) into RGB camera 3D coordinates (origin = Camera lens).
Includes batch processing to transform an entire dataset in-place or into a
new directory.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from dataset_utils import (
    load_dataset_pairs,
    load_joint_json,
    save_leap_json,
    joints_dict_to_array,
)
from calibrate_intrinsics import load_intrinsics
from calibrate_extrinsics import load_extrinsics


def apply_extrinsic(joints_3d: np.ndarray, rvec: np.ndarray,
                    tvec: np.ndarray) -> np.ndarray:
    """
    Transform 3D joints from Leap space to Camera space.

    Args:
        joints_3d: (N, 3) array in Leap space (mm).
        rvec: (3, 1) rotation vector from solvePnP.
        tvec: (3, 1) translation vector from solvePnP.

    Returns:
        (N, 3) array in Camera space (mm).
    """
    if len(joints_3d) == 0:
        return joints_3d

    R, _ = cv2.Rodrigues(rvec)
    # Transform: P_cam = R * P_leap + t
    # For (N, 3) we do: (R @ joints_3d.T).T + t.T
    transformed = (R @ joints_3d.T).T + tvec.T
    return transformed


def array_to_joints_dict(joints_array: np.ndarray) -> dict:
    """Convert a (21, 3) array back to the labelled JSON dictionary format."""
    joints_dict = {}
    for idx, name in config.JOINT_NAMES.items():
        joints_dict[f"{idx}_{name}"] = {
            "x": float(joints_array[idx, 0]),
            "y": float(joints_array[idx, 1]),
            "z": float(joints_array[idx, 2]),
        }
    return joints_dict


def transform_dataset(dataset_dir: Path, output_dir: Path, rvec: np.ndarray,
                      tvec: np.ndarray):
    """
    Batch transform all joint JSONs in a dataset to the camera coordinate space.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    pairs = load_dataset_pairs(dataset_dir)

    if not pairs:
        print(f"[ERROR] No valid frame pairs found in {dataset_dir}")
        sys.exit(1)

    print(f"[TRANSFORM] Processing {len(pairs)} frames...")

    out_leap_dir = output_dir / "leap"
    out_leap_dir.mkdir(parents=True, exist_ok=True)

    for rgb_path, leap_path in pairs:
        # Load
        data = load_joint_json(leap_path)
        joints_arr = joints_dict_to_array(data["joints"])

        # Transform
        transformed_arr = apply_extrinsic(joints_arr, rvec, tvec)

        # Convert back
        new_joints = array_to_joints_dict(transformed_arr)

        # Save
        save_leap_json(
            out_leap_dir,
            data["frame_id"],
            new_joints,
            data["hand_type"],
            data["timestamp_rgb"],
            data["timestamp_leap"]
        )

        # Copy RGB file if output dir is different
        if dataset_dir != output_dir:
            out_rgb_dir = output_dir / "rgb"
            out_rgb_dir.mkdir(parents=True, exist_ok=True)
            out_rgb_path = out_rgb_dir / rgb_path.name
            if not out_rgb_path.exists():
                import shutil
                shutil.copy2(rgb_path, out_rgb_path)

    # Update metadata
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["extrinsic_calibrated"] = True
        
        out_meta = output_dir / "metadata.json"
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"[TRANSFORM] Complete. Transformed data saved to {output_dir}")


def project_to_2d(joints_3d_cam: np.ndarray, K: np.ndarray,
                  dist: np.ndarray) -> np.ndarray:
    """
    Project 3D camera-space joints onto 2D image pixels.

    Args:
        joints_3d_cam: (N, 3) array in Camera space (mm).
        K: (3, 3) camera intrinsic matrix.
        dist: distortion coefficients.

    Returns:
        (N, 2) array of pixel coordinates.
    """
    # cv2.projectPoints expects (N, 1, 3) or (N, 3)
    # We pass zero rvec/tvec because the points are ALREADY in camera space
    rvec_zero = np.zeros((3, 1))
    tvec_zero = np.zeros((3, 1))
    
    img_points, _ = cv2.projectPoints(joints_3d_cam, rvec_zero, tvec_zero, K, dist)
    return img_points.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3c: Apply extrinsic transformation to a dataset."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Input dataset directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to overwriting input)")
    parser.add_argument("--extrinsics", type=str,
                        default=str(config.CALIBRATION_DIR / "extrinsic_calibration.json"),
                        help="Path to extrinsic_calibration.json")
    args = parser.parse_args()

    extrinsics_path = Path(args.extrinsics)
    if not extrinsics_path.exists():
        print(f"[ERROR] Extrinsics file not found: {extrinsics_path}")
        sys.exit(1)

    rvec, tvec, R = load_extrinsics(extrinsics_path)
    print(f"[TRANSFORM] Loaded extrinsics.")
    print(f"  tvec: {tvec.flatten()} mm")

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output) if args.output else dataset_dir

    transform_dataset(dataset_dir, output_dir, rvec, tvec)


if __name__ == "__main__":
    main()
