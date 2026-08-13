"""
visualize.py — Utility to visualize joints projected onto RGB frames.

Useful for verifying synchronization and extrinsic calibration.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from dataset_utils import load_dataset_pairs, load_joint_json, joints_dict_to_array
from calibrate_intrinsics import load_intrinsics
from transform import project_to_2d


def draw_skeleton(img: np.ndarray, points_2d: np.ndarray, color=(0, 255, 0), thickness=2):
    """Draw the 21-joint skeleton on an image."""
    # Draw bones
    for joint_a, joint_b in config.SKELETON_CONNECTIONS:
        # Check if points are valid (e.g., not NaN and within reasonable bounds)
        pt_a = points_2d[joint_a]
        pt_b = points_2d[joint_b]
        
        if np.isnan(pt_a).any() or np.isnan(pt_b).any():
            continue
            
        # Protect against massive coordinates overflowing OpenCV's int types
        if np.abs(pt_a).max() > 1e5 or np.abs(pt_b).max() > 1e5:
            continue
            
        p1 = (int(pt_a[0]), int(pt_a[1]))
        p2 = (int(pt_b[0]), int(pt_b[1]))
        
        cv2.line(img, p1, p2, color, thickness)
        
    # Draw joints
    for i, pt in enumerate(points_2d):
        if np.isnan(pt).any() or np.abs(pt).max() > 1e5:
            continue
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(img, p, thickness + 2, (0, 0, 255), -1)


def visualize_dataset(dataset_dir: Path, K: np.ndarray, dist: np.ndarray, 
                      delay_ms: int = 30):
    """Play back a dataset with projected joints."""
    pairs = load_dataset_pairs(dataset_dir)
    if not pairs:
        print("[ERROR] No valid frame pairs found in dataset.")
        sys.exit(1)
        
    print(f"[VISUALIZE] Playing back {len(pairs)} frames from {dataset_dir}")
    print("  Press 'q' to quit, 'space' to pause/resume.")
    
    paused = False
    idx = 0
    
    while idx < len(pairs):
        rgb_path, leap_path = pairs[idx]
        
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            idx += 1
            continue
            
        # Load 3D joints (assumed to be already transformed to Camera space)
        data = load_joint_json(leap_path)
        joints_3d = joints_dict_to_array(data["joints"])
        
        # Project to 2D
        points_2d = project_to_2d(joints_3d, K, dist)
        
        # Draw
        display = frame.copy()
        draw_skeleton(display, points_2d)
        
        # Meta info
        cv2.putText(display, f"Frame: {data['frame_id']} | dt: {data['sync_delta_ms']}ms", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        cv2.imshow("Dataset Visualization", display)
        
        wait_time = 0 if paused else delay_ms
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            
        if not paused:
            idx += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize projected dataset joints")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--intrinsics", type=str,
                        default=str(config.CALIBRATION_DIR / "camera_intrinsics.json"))
    parser.add_argument("--delay", type=int, default=30, help="Playback delay ms")
    args = parser.parse_args()

    intrinsics_path = Path(args.intrinsics)
    if not intrinsics_path.exists():
        print("[ERROR] Intrinsics required for projection.")
        sys.exit(1)
        
    K, dist = load_intrinsics(intrinsics_path)
    
    visualize_dataset(Path(args.dataset), K, dist, args.delay)


if __name__ == "__main__":
    main()
