"""
dataset_utils.py — Stage 2: Dataset generation utilities.

Handles directory creation, frame saving (RGB JPEGs + Leap JSON), and session
metadata generation. Used by capture.py during recording and by other scripts
for loading/validating datasets.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config


# ==============================================================================
# Directory Management
# ==============================================================================

def create_dataset_dirs(output_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Create the dataset directory structure.

    Returns:
        (dataset_root, rgb_dir, leap_dir)
    """
    root = Path(output_dir) if output_dir else config.DATASET_DIR
    rgb_dir = root / "rgb"
    leap_dir = root / "leap"

    root.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    leap_dir.mkdir(parents=True, exist_ok=True)

    return root, rgb_dir, leap_dir


# ==============================================================================
# Frame Saving
# ==============================================================================

def frame_filename(frame_id: int) -> str:
    """Generate zero-padded frame filename base: frame_0001, frame_0002, …"""
    return f"frame_{frame_id:04d}"


def save_rgb_frame(rgb_dir: Path, frame_id: int, frame: np.ndarray,
                   quality: int = 95) -> Path:
    """
    Save an RGB frame as a JPEG image.

    Args:
        rgb_dir: Directory to save into.
        frame_id: Sequential frame number.
        frame: BGR numpy array from OpenCV.
        quality: JPEG quality 0–100.

    Returns:
        Path to the saved file.
    """
    filename = frame_filename(frame_id) + ".jpg"
    filepath = rgb_dir / filename
    cv2.imwrite(str(filepath), frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    return filepath


def save_leap_json(leap_dir: Path, frame_id: int,
                   joints: Dict[str, Dict[str, float]],
                   hand_type: str,
                   timestamp_rgb: float,
                   timestamp_leap: float) -> Path:
    """
    Save LeapC joint data as a labelled JSON file.

    JSON structure:
    {
        "frame_id": 1,
        "timestamp_rgb": 1691234567.123,
        "timestamp_leap": 1691234567.130,
        "sync_delta_ms": 7.0,
        "hand_type": "Right",
        "joints": {
            "0_Wrist": {"x": 12.3, "y": 45.6, "z": -78.9},
            …
            "20_Pinky_Tip": {"x": …, "y": …, "z": …}
        }
    }

    Args:
        leap_dir: Directory to save into.
        frame_id: Sequential frame number.
        joints: Dict mapping "idx_Name" → {"x", "y", "z"} (mm).
        hand_type: "Left" or "Right".
        timestamp_rgb: Wall-clock timestamp of the paired RGB frame.
        timestamp_leap: Wall-clock timestamp of the Leap tracking event.

    Returns:
        Path to the saved file.
    """
    filename = frame_filename(frame_id) + ".json"
    filepath = leap_dir / filename

    delta_ms = abs(timestamp_rgb - timestamp_leap) * 1000.0

    data = {
        "frame_id": frame_id,
        "timestamp_rgb": timestamp_rgb,
        "timestamp_leap": timestamp_leap,
        "sync_delta_ms": round(delta_ms, 3),
        "hand_type": hand_type,
        "joints": joints,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


# ==============================================================================
# Session Metadata
# ==============================================================================

def save_session_metadata(dataset_root: Path, total_frames: int,
                          camera_index: int, camera_resolution: Tuple[int, int],
                          duration_sec: float, sync_threshold: float,
                          calibrated: bool = False,
                          extra: Optional[dict] = None) -> Path:
    """
    Save session-level metadata to metadata.json.

    Args:
        dataset_root: Root of the dataset directory.
        total_frames: Number of synchronized pairs captured.
        camera_index: OpenCV camera device index.
        camera_resolution: (width, height) of captured RGB frames.
        duration_sec: Total capture duration in seconds.
        sync_threshold: Timestamp sync threshold used (seconds).
        calibrated: Whether extrinsic calibration was applied.
        extra: Any additional key-value pairs to store.

    Returns:
        Path to the saved metadata file.
    """
    filepath = dataset_root / "metadata.json"

    meta = {
        "capture_date": datetime.datetime.now().isoformat(),
        "total_frames": total_frames,
        "camera_index": camera_index,
        "camera_resolution": list(camera_resolution),
        "capture_duration_sec": round(duration_sec, 2),
        "sync_threshold_sec": sync_threshold,
        "extrinsic_calibrated": calibrated,
    }

    if extra:
        meta.update(extra)

    with open(filepath, "w") as f:
        json.dump(meta, f, indent=2)

    return filepath


# ==============================================================================
# Dataset Loading (for downstream stages)
# ==============================================================================

def load_joint_json(filepath: Path) -> Dict:
    """Load a single frame's joint JSON and return the parsed dict."""
    with open(filepath, "r") as f:
        return json.load(f)


def joints_dict_to_array(joints: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Convert the labelled joints dict to a (21, 3) numpy array.

    The keys are expected in the form "idx_Name" and are sorted by idx.
    """
    arr = np.zeros((config.NUM_JOINTS, 3), dtype=np.float32)
    for key, xyz in joints.items():
        idx = int(key.split("_")[0])
        arr[idx] = [xyz["x"], xyz["y"], xyz["z"]]
    return arr


def load_dataset_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """
    Discover all matching (rgb, leap) file pairs in a dataset directory.

    Returns:
        Sorted list of (rgb_path, leap_path) tuples.
    """
    rgb_dir = Path(dataset_root) / "rgb"
    leap_dir = Path(dataset_root) / "leap"

    rgb_files = sorted(rgb_dir.glob("frame_*.jpg"))
    pairs = []

    for rgb_path in rgb_files:
        stem = rgb_path.stem  # e.g. "frame_0001"
        leap_path = leap_dir / (stem + ".json")
        if leap_path.exists():
            pairs.append((rgb_path, leap_path))

    return pairs


def load_all_joints(dataset_root: Path) -> np.ndarray:
    """
    Load all joint arrays from a dataset.

    Returns:
        (N, 21, 3) numpy array where N is the number of frames.
    """
    pairs = load_dataset_pairs(dataset_root)
    all_joints = []
    for _, leap_path in pairs:
        data = load_joint_json(leap_path)
        arr = joints_dict_to_array(data["joints"])
        all_joints.append(arr)
    return np.array(all_joints, dtype=np.float32)
