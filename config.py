"""
config.py — Central configuration for the 3Di Evaluation Framework.

All shared constants, paths, joint mappings, and default parameters live here
so every script in the pipeline references a single source of truth.
"""

import os
from pathlib import Path

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RGB_DIR = DATASET_DIR / "rgb"
LEAP_DIR = DATASET_DIR / "leap"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_IR_DIR = PROJECT_ROOT / "dataset_ir"

# ==============================================================================
# 3DFist (stereo IR)
# ==============================================================================
IR_INPUT_SIZE = 128            # Per-view hand crop fed to Stereo3DFistNet
# Written into every 3DFist checkpoint. Bump when preprocessing or label space
# changes so stale checkpoints are rejected instead of silently mispredicting.
FIST_INPUT_FORMAT = "stereo_ir_perspective_v3"
HANDLOC_INPUT_FORMAT = "ir_handloc_v1"

# ==============================================================================
# Camera
# ==============================================================================
CAMERA_INDEX = 1               # OpenCV VideoCapture device index (1 = Dell RGB webcam)
CAMERA_WIDTH = 640             # Requested resolution (may fall back to native)
CAMERA_HEIGHT = 480
CAMERA_FPS = 30                # Requested FPS

# ==============================================================================
# Synchronization (Stage 1)
# ==============================================================================
SYNC_THRESHOLD_SEC = 0.015     # 15 ms — max allowable timestamp delta
MIN_SAVE_INTERVAL_SEC = 0.033  # ~30 FPS cap on saved pairs (debounce)
LEAP_POLL_TIMEOUT_MS = 100     # LeapC poll timeout per cycle
CAPTURE_DURATION_SEC = 60      # Default recording duration

# ==============================================================================
# Calibration (Stage 3)
# ==============================================================================
CHECKERBOARD_ROWS = 6          # Inner corner rows
CHECKERBOARD_COLS = 9          # Inner corner columns
CHECKERBOARD_SQUARE_MM = 25.0  # Physical square edge length in millimetres

# ==============================================================================
# Evaluation (Stage 4)
# ==============================================================================
NUM_JOINTS = 21
WRIST_JOINT_INDEX = 0          # Root joint for alignment

# ==============================================================================
# Training (Stage 5)
# ==============================================================================
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-5
IMAGE_SIZE = 224               # Input size for ViT models
TRAIN_SPLIT = 0.8              # 80 / 20 train–val split

# ImageNet normalisation (standard for ViT pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==============================================================================
# 21-Joint Hand Model — Name Mapping
# ==============================================================================
# This follows the standard hand keypoint convention used by MediaPipe, MANO,
# and Ultraleap.  The LeapC bone-traversal order that produces these joints is
# documented in the extraction function inside capture.py.
#
#   Wrist  ──▶  Thumb (4 joints)  ──▶  Index (4)  ──▶  Middle (4)
#          ──▶  Ring  (4 joints)  ──▶  Pinky (4)
# ==============================================================================

JOINT_NAMES = {
    0:  "Wrist",
    1:  "Thumb_CMC",
    2:  "Thumb_MCP",
    3:  "Thumb_IP",
    4:  "Thumb_Tip",
    5:  "Index_MCP",
    6:  "Index_PIP",
    7:  "Index_DIP",
    8:  "Index_Tip",
    9:  "Middle_MCP",
    10: "Middle_PIP",
    11: "Middle_DIP",
    12: "Middle_Tip",
    13: "Ring_MCP",
    14: "Ring_PIP",
    15: "Ring_DIP",
    16: "Ring_Tip",
    17: "Pinky_MCP",
    18: "Pinky_PIP",
    19: "Pinky_DIP",
    20: "Pinky_Tip",
}

# Skeleton bone connections for drawing — each tuple is (parent_idx, child_idx)
SKELETON_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Per-finger joint ranges (for per-finger error analysis)
FINGER_RANGES = {
    "Thumb":  (1, 4),
    "Index":  (5, 8),
    "Middle": (9, 12),
    "Ring":   (13, 16),
    "Pinky":  (17, 20),
}


def ensure_dirs():
    """Create all required output directories if they don't exist."""
    for d in [DATASET_DIR, RGB_DIR, LEAP_DIR, CALIBRATION_DIR,
              CHECKPOINTS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
