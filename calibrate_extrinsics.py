"""
calibrate_extrinsics.py — Stage 3b: Extrinsic Spatial Calibration.

Aligns the Ultraleap 3Di coordinate space to the RGB camera coordinate space
using a physical checkerboard. The user touches checkerboard corners with a
Leap-tracked fingertip to provide 3D↔2D correspondences, then cv2.solvePnP
computes the extrinsic transformation (rvec, tvec).

Usage:
    python calibrate_extrinsics.py --camera 0 \\
        --intrinsics calibration/camera_intrinsics.json
"""

import argparse
import json
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    import leap
except ImportError:
    leap = None

import config
from calibrate_intrinsics import find_checkerboard, load_intrinsics

# Attempt to reuse capture utilities
try:
    from capture import (
        extract_21_joints, get_hand_type_str, SharedState,
        LeapTrackingListener, generate_dummy_joints,
    )
except ImportError:
    pass


# ==============================================================================
# Extrinsic Calibration
# ==============================================================================

def interactive_corner_registration(camera_index: int, K: np.ndarray,
                                    dist: np.ndarray, rows: int, cols: int,
                                    square_mm: float,
                                    use_simulated: bool = False):
    """
    Interactive workflow to register 3D Leap coordinates to 2D checkerboard
    corners.

    Steps:
        1. Detect the checkerboard in the RGB feed → 2D image points.
        2. For each corner, user touches it with their index fingertip.
        3. User presses SPACE to record the Leap fingertip 3D position.
        4. Repeat for all corners (or a subset).

    Returns:
        (obj_points_3d, img_points_2d) as numpy arrays.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    # ---- Detect checkerboard first ----
    print("[EXTRINSIC] Show the checkerboard to the camera.")
    print("  Press SPACE when it's detected to lock the 2D corners.\n")

    locked_corners = None
    locked_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        found, corners = find_checkerboard(frame, rows, cols)

        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(display, "Checkerboard DETECTED — SPACE to lock",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        else:
            cv2.putText(display, "Searching for checkerboard...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow("Extrinsic Calibration", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" ") and found:
            locked_corners = corners.reshape(-1, 2)
            locked_frame = frame.copy()
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    total_corners = rows * cols
    print(f"\n[EXTRINSIC] Locked {total_corners} 2D corners.")
    print(f"  Now touch each numbered corner with your index fingertip.")
    print(f"  Press SPACE to record each 3D point. Press S to skip a corner.\n")

    # ---- Set up Leap tracking ----
    state = SharedState()

    if leap and not use_simulated:
        connection = leap.Connection()
        listener = LeapTrackingListener(state, hand_filter="any")
        connection.add_listener(listener)
        ctx = connection.open()
        ctx.__enter__()
        time.sleep(0.5)
    else:
        # Start simulated producer in background
        def sim_producer():
            while state.running:
                ts = time.time()
                joints = generate_dummy_joints()
                with state.lock:
                    state.leap_joints = joints
                    state.leap_timestamp = ts
                    state.leap_hand_type = "Right"
                time.sleep(1.0 / 90)

        t = threading.Thread(target=sim_producer, daemon=True)
        t.start()
        time.sleep(0.3)

    # ---- Register each corner ----
    registered_2d = []
    registered_3d = []
    corner_idx = 0

    while corner_idx < total_corners:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # Draw all corners with numbers
        for i, (cx, cy) in enumerate(locked_corners):
            color = (0, 255, 0) if i < corner_idx else (
                (0, 255, 255) if i == corner_idx else (128, 128, 128))
            cv2.circle(display, (int(cx), int(cy)), 6, color, -1)
            cv2.putText(display, str(i), (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Highlight current target
        tx, ty = locked_corners[corner_idx]
        cv2.circle(display, (int(tx), int(ty)), 12, (0, 0, 255), 3)

        # Get current Leap fingertip position (Index tip = joint 8)
        with state.lock:
            joints = state.leap_joints

        tip_text = "No hand detected"
        if joints:
            # Index tip is key "8_Index_Tip"
            tip_key = f"8_{config.JOINT_NAMES[8]}"
            if tip_key in joints:
                tip = joints[tip_key]
                tip_text = (f"Fingertip: ({tip['x']:.1f}, "
                            f"{tip['y']:.1f}, {tip['z']:.1f}) mm")

        cv2.putText(display, f"Touch corner #{corner_idx} — SPACE to record",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, tip_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(display, f"Registered: {len(registered_2d)}/{total_corners}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Extrinsic Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" ") and joints:
            tip_key = f"8_{config.JOINT_NAMES[8]}"
            if tip_key in joints:
                tip = joints[tip_key]
                registered_3d.append([tip["x"], tip["y"], tip["z"]])
                registered_2d.append([tx, ty])
                print(f"  ✓ Corner #{corner_idx}: 3D=({tip['x']:.1f}, "
                      f"{tip['y']:.1f}, {tip['z']:.1f})  "
                      f"2D=({tx:.1f}, {ty:.1f})")
                corner_idx += 1
        elif key == ord("s"):
            print(f"  → Skipped corner #{corner_idx}")
            corner_idx += 1
        elif key == ord("q"):
            break

    # Cleanup
    state.running = False
    cap.release()
    cv2.destroyAllWindows()

    if leap and not use_simulated:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass

    obj_3d = np.array(registered_3d, dtype=np.float64)
    img_2d = np.array(registered_2d, dtype=np.float64)

    print(f"\n[EXTRINSIC] Registered {len(registered_2d)} point correspondences.")
    return obj_3d, img_2d


def compute_extrinsics(obj_3d: np.ndarray, img_2d: np.ndarray,
                       K: np.ndarray, dist: np.ndarray):
    """
    Compute the extrinsic transformation using cv2.solvePnP.

    Args:
        obj_3d: (N, 3) array of 3D points in Leap coordinate space (mm).
        img_2d: (N, 2) array of corresponding 2D image points (pixels).
        K: Camera intrinsic matrix (3×3).
        dist: Distortion coefficients.

    Returns:
        (rvec, tvec, R, reprojection_error)
    """
    if len(obj_3d) < 4:
        print("[ERROR] Need at least 4 point correspondences for solvePnP.")
        sys.exit(1)

    success, rvec, tvec = cv2.solvePnP(
        obj_3d, img_2d.reshape(-1, 1, 2), K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("[ERROR] solvePnP failed.")
        sys.exit(1)

    # Rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Reprojection error
    projected, _ = cv2.projectPoints(obj_3d, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img_2d, axis=1)
    mean_error = float(np.mean(errors))

    return rvec, tvec, R, mean_error


def save_extrinsics(output_path: Path, rvec: np.ndarray, tvec: np.ndarray,
                    R: np.ndarray, reprojection_error: float):
    """Save the extrinsic calibration to a JSON file."""
    data = {
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "rotation_matrix": R.tolist(),
        "mean_reprojection_error_px": round(reprojection_error, 4),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[EXTRINSIC] Saved to {output_path}")


def load_extrinsics(filepath: Path):
    """Load extrinsic calibration from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    rvec = np.array(data["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(data["tvec"], dtype=np.float64).reshape(3, 1)
    R = np.array(data["rotation_matrix"], dtype=np.float64)
    return rvec, tvec, R


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3b: Extrinsic calibration (Leap → Camera)."
    )
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--intrinsics", type=str,
                        default=str(config.CALIBRATION_DIR / "camera_intrinsics.json"),
                        help="Path to camera_intrinsics.json")
    parser.add_argument("--rows", type=int, default=config.CHECKERBOARD_ROWS)
    parser.add_argument("--cols", type=int, default=config.CHECKERBOARD_COLS)
    parser.add_argument("--square-mm", type=float,
                        default=config.CHECKERBOARD_SQUARE_MM)
    parser.add_argument("--output", type=str,
                        default=str(config.CALIBRATION_DIR / "extrinsic_calibration.json"))
    parser.add_argument("--simulate-leap", action="store_true",
                        help="Use simulated Leap data")
    args = parser.parse_args()

    # Load intrinsics
    intrinsics_path = Path(args.intrinsics)
    if not intrinsics_path.exists():
        print(f"[ERROR] Intrinsics file not found: {intrinsics_path}")
        print("  Run calibrate_intrinsics.py first.")
        sys.exit(1)

    K, dist = load_intrinsics(intrinsics_path)
    print(f"[EXTRINSIC] Loaded intrinsics (RMS error from intrinsic calib)")

    # Interactive registration
    use_sim = args.simulate_leap or (leap is None)
    obj_3d, img_2d = interactive_corner_registration(
        args.camera, K, dist, args.rows, args.cols, args.square_mm,
        use_simulated=use_sim
    )

    if len(obj_3d) < 4:
        print("[ERROR] Not enough correspondences. Need at least 4.")
        sys.exit(1)

    # Compute extrinsics
    print("\n[EXTRINSIC] Computing transformation with solvePnP...")
    rvec, tvec, R, reproj_err = compute_extrinsics(obj_3d, img_2d, K, dist)

    print(f"\n  Rotation Vector:  {rvec.flatten()}")
    print(f"  Translation (mm): {tvec.flatten()}")
    print(f"  Mean Reprojection Error: {reproj_err:.3f} px")

    if reproj_err > 5.0:
        print(f"\n  [WARNING] Reprojection error is high (> 5px).")
        print(f"  Consider re-calibrating with more careful corner touches.")

    # Save
    save_extrinsics(Path(args.output), rvec, tvec, R, reproj_err)


if __name__ == "__main__":
    main()
