"""
calibrate_intrinsics.py — Stage 3a: RGB Camera Intrinsic Calibration.

Captures or loads multiple views of a printed checkerboard to compute the
camera matrix K and distortion coefficients. These are required before
extrinsic calibration can be performed.

Usage:
    # Live capture mode — capture checkerboard views interactively
    python calibrate_intrinsics.py --camera 0 --num-views 15

    # From pre-saved images
    python calibrate_intrinsics.py --images calibration/intrinsic_images/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import config


def find_checkerboard(frame: np.ndarray, rows: int, cols: int):
    """
    Detect checkerboard corners in a frame.

    Returns:
        (found, corners) where corners is refined sub-pixel positions
        or None if not found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (cols, rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def capture_views_live(camera_index: int, rows: int, cols: int,
                       num_views: int) -> list:
    """
    Interactively capture checkerboard views from a live camera feed.

    Press SPACE to capture a view when the checkerboard is detected.
    Press Q to quit early.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        sys.exit(1)

    print(f"[INTRINSIC] Capturing {num_views} checkerboard views.")
    print("  Press SPACE when checkerboard is visible to capture.")
    print("  Press Q to finish early.\n")

    captured_corners = []
    captured_images = []

    while len(captured_corners) < num_views:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        found, corners = find_checkerboard(frame, rows, cols)

        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(display, "DETECTED — Press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        else:
            cv2.putText(display, "Searching for checkerboard...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        count_text = f"Views: {len(captured_corners)}/{num_views}"
        cv2.putText(display, count_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Intrinsic Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" ") and found:
            captured_corners.append(corners)
            captured_images.append(frame.copy())
            print(f"  ✓ Captured view {len(captured_corners)}/{num_views}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_corners, captured_images


def load_views_from_dir(image_dir: Path, rows: int, cols: int):
    """Load checkerboard views from pre-saved images in a directory."""
    image_dir = Path(image_dir)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(sorted(image_dir.glob(ext)))

    if not image_paths:
        print(f"[ERROR] No images found in {image_dir}")
        sys.exit(1)

    print(f"[INTRINSIC] Loading {len(image_paths)} images from {image_dir}")

    captured_corners = []
    captured_images = []

    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        found, corners = find_checkerboard(frame, rows, cols)
        if found:
            captured_corners.append(corners)
            captured_images.append(frame)
            print(f"  ✓ Detected in {path.name}")
        else:
            print(f"  ✗ Not found in {path.name}")

    return captured_corners, captured_images


def calibrate(captured_corners: list, image_shape: tuple,
              rows: int, cols: int, square_mm: float):
    """
    Run OpenCV camera calibration.

    Args:
        captured_corners: List of (N, 1, 2) corner arrays.
        image_shape: (height, width) of the images.
        rows, cols: Inner checkerboard corner dimensions.
        square_mm: Physical size of each square in mm.

    Returns:
        (camera_matrix, dist_coeffs, rvecs, tvecs, rms_error)
    """
    # Prepare object points: (0,0,0), (25,0,0), (50,0,0), ...
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    obj_points = [objp] * len(captured_corners)  # Same for every view
    img_points = captured_corners

    h, w = image_shape[:2]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    return camera_matrix, dist_coeffs, rvecs, tvecs, rms


def save_intrinsics(output_path: Path, camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray, rms_error: float,
                    image_size: tuple):
    """Save camera intrinsics to a JSON file."""
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rms_reprojection_error": round(rms_error, 6),
        "image_size": list(image_size),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[INTRINSIC] Saved to {output_path}")
    return data


def load_intrinsics(filepath: Path):
    """Load camera intrinsics from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3a: Intrinsic camera calibration via checkerboard."
    )
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Camera device index for live capture")
    parser.add_argument("--images", type=str, default=None,
                        help="Directory of pre-saved checkerboard images")
    parser.add_argument("--num-views", type=int, default=15,
                        help="Number of views to capture (live mode)")
    parser.add_argument("--rows", type=int, default=config.CHECKERBOARD_ROWS)
    parser.add_argument("--cols", type=int, default=config.CHECKERBOARD_COLS)
    parser.add_argument("--square-mm", type=float,
                        default=config.CHECKERBOARD_SQUARE_MM)
    parser.add_argument("--output", type=str,
                        default=str(config.CALIBRATION_DIR / "camera_intrinsics.json"))
    args = parser.parse_args()

    # Capture or load views
    if args.images:
        corners, images = load_views_from_dir(
            Path(args.images), args.rows, args.cols)
    else:
        corners, images = capture_views_live(
            args.camera, args.rows, args.cols, args.num_views)

    if len(corners) < 3:
        print(f"[ERROR] Need at least 3 views, got {len(corners)}. Aborting.")
        sys.exit(1)

    print(f"\n[INTRINSIC] Calibrating with {len(corners)} views...")

    # Calibrate
    image_shape = images[0].shape
    K, dist, rvecs, tvecs, rms = calibrate(
        corners, image_shape, args.rows, args.cols, args.square_mm)

    print(f"\n  RMS Reprojection Error: {rms:.4f} pixels")
    print(f"  Camera Matrix K:")
    print(f"    fx = {K[0, 0]:.2f}  fy = {K[1, 1]:.2f}")
    print(f"    cx = {K[0, 2]:.2f}  cy = {K[1, 2]:.2f}")

    # Save
    save_intrinsics(
        Path(args.output), K, dist, rms,
        (image_shape[1], image_shape[0])
    )


if __name__ == "__main__":
    main()
