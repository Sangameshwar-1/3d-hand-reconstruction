"""
fist_dataset.py — PyTorch Dataset for Dual-IR 3D Hand Pose.

Loads:
  - Left active IR image (raw 384x384 grayscale from the 3Di).
  - Right active IR image (raw 384x384 grayscale from the 3Di).
  - Paired 21-joint 3D coordinates in millimetres (native LeapC tracking space,
    root-aligned to the wrist).

Layout produced by capture_ir.py (one folder per recording session):

  dataset_ir/
  ├── session_YYYYmmdd_HHMMSS/
  │   ├── left/frame_0001.png
  │   ├── right/frame_0001.png
  │   ├── leap/frame_0001.json
  │   └── metadata.json
  └── ...

A flat dataset_ir/{left,right,leap} layout is also accepted as a single session.

Frames from the same session are near-duplicates (the 3Di runs at ~90 FPS),
so the train/val split is made by session whenever there are 2+ sessions.
"""

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_utils import load_joint_json, joints_dict_to_array
from ir_geometry import bbox_from_pixels, load_session_luts
from ir_transforms import StereoPerspectivePreprocessor
from perspective_crop import hand_center_from_boxes


def find_sessions(dataset_dir: Path) -> List[Path]:
    """Session folders (each containing left/, right/, leap/) under dataset_dir."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return []
    sessions = sorted(p for p in dataset_dir.iterdir() if (p / "left").is_dir())
    if (dataset_dir / "left").is_dir():
        sessions.insert(0, dataset_dir)  # flat legacy layout counts as one session
    return sessions


def find_stereo_ir_samples(session_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """Matching (left_png, right_png, leap_json) triplets in one session."""
    samples = []
    for l_path in sorted((session_dir / "left").glob("*.png")):
        r_path = session_dir / "right" / l_path.name
        j_path = session_dir / "leap" / f"{l_path.stem}.json"
        if r_path.exists() and j_path.exists():
            samples.append((l_path, r_path, j_path))
    return samples


def split_sessions(sessions: List[Path], split: str, val_ratio: float, dataset_dir: Path):
    """Returns [(session, frame_slice)] for the requested split.

    dataset_dir/val_sessions.txt, if present, names the held-out sessions (one per line).
    """
    val_list = dataset_dir / "val_sessions.txt"
    if val_list.exists():
        names = set(val_list.read_text().split())
        return [(s, slice(None)) for s in sessions if (s.name in names) == (split != "train")]
    if len(sessions) >= 2:
        n_val = max(1, int(round(len(sessions) * val_ratio)))
        chosen = sessions[-n_val:] if split != "train" else sessions[:-n_val]
        return [(s, slice(None)) for s in chosen]

    # Single session: contiguous split with a gap so neighbouring frames
    # don't leak across the boundary.
    n = len(find_stereo_ir_samples(sessions[0]))
    cut = int(n * (1.0 - val_ratio))
    gap = min(30, n // 20)
    if split == "train":
        return [(sessions[0], slice(0, max(0, cut - gap)))]
    return [(sessions[0], slice(cut, None))]


def load_split_frames(dataset_dir, split: str, val_ratio: float, min_confidence: float = 0.0,
                      with_meta: bool = False):
    """
    Load raw frames of one split into memory.

    Returns a list of (img_l, img_r, joints_mm, bbox_l, bbox_r, luts): joints
    are in native LeapC space (not root-aligned); boxes come from projecting
    the joints into each view with the session's factory calibration
    (ir_geometry); luts is that session's per-camera RectilinearLUT dict.
    With with_meta, returns (frames, metas): one {"is_right", "grab"} per frame.
    """
    dataset_dir = Path(dataset_dir)
    sessions = find_sessions(dataset_dir)
    if not sessions or not any(find_stereo_ir_samples(s) for s in sessions):
        raise FileNotFoundError(
            f"No stereo IR samples (left/*.png + right/*.png + leap/*.json) in {dataset_dir}. "
            f"Record data with: python capture_ir.py --output-dir {dataset_dir}"
        )

    frames = []
    metas = []
    skipped = {"no_calibration": 0, "hand_not_in_view": 0, "low_confidence": 0, "identical_views": 0,
               "no_triangulation": 0, "ultraleap_wrong_hand": 0, "excluded": 0}
    chosen = split_sessions(sessions, split, val_ratio, dataset_dir)
    # session/frame stems whose Ultraleap hand a trained model disputes (sessions without a named hand)
    excl_file = dataset_dir / "excluded_frames.txt"
    excluded = set(excl_file.read_text().split()) if excl_file.exists() else set()
    for session, sl in chosen:
        samples = find_stereo_ir_samples(session)[sl]
        luts = load_session_luts(session)
        if luts is None:
            # Recorded before capture_ir.py saved rect_lut.npz / calibration.json
            skipped["no_calibration"] += len(samples)
            continue
        for l_path, r_path, j_path in samples:
            joint_data = load_joint_json(j_path)
            if joint_data.get("confidence", 1.0) < min_confidence:
                skipped["low_confidence"] += 1
                continue
            # Ultraleap often tracks the right palm as a left back (same silhouette); when the
            # recording names the hand, its label and its mirrored-chirality skeleton are wrong.
            named = joint_data.get("pose_label", "").split("_")[0].capitalize()
            if named in ("Left", "Right"):
                if joint_data.get("hand_type") != named:
                    skipped["ultraleap_wrong_hand"] += 1
                    continue
            elif f"{session.name}/{j_path.stem}" in excluded:
                skipped["excluded"] += 1
                continue
            img_l = cv2.imread(str(l_path), cv2.IMREAD_GRAYSCALE)
            img_r = cv2.imread(str(r_path), cv2.IMREAD_GRAYSCALE)
            if img_l is None or img_r is None:
                continue
            if np.array_equal(img_l, img_r):
                # Recorded by the old capture that ignored LeapC's image offset
                skipped["identical_views"] += 1
                continue
            joints = joints_dict_to_array(joint_data["joints"])  # (21, 3) mm
            bbox_l = bbox_from_pixels(luts["stereo_left"].project(joints), img_l.shape)
            bbox_r = bbox_from_pixels(luts["stereo_right"].project(joints), img_r.shape)
            if bbox_l is None or bbox_r is None:
                skipped["hand_not_in_view"] += 1
                continue
            if hand_center_from_boxes(luts, bbox_l, bbox_r) is None:
                skipped["no_triangulation"] += 1
                continue
            frames.append((img_l, img_r, joints, bbox_l, bbox_r, luts))
            metas.append({"is_right": float(joint_data.get("hand_type", "Right") == "Right"),
                          "grab": float(joint_data.get("grab_strength", 0.0))})

    print(f"[DATASET] '{split}': {len(frames)} frames from {len(chosen)} session(s); skipped {skipped}")
    if skipped["identical_views"] or skipped["no_calibration"]:
        print("[DATASET] WARNING: some frames come from captures made before the stereo/calibration "
              "fixes (identical views or no rect_lut.npz) and cannot be used; re-record them.")
    if not frames:
        raise RuntimeError(f"No usable samples for split '{split}' in {dataset_dir}.")
    return (frames, metas) if with_meta else frames


_MIRROR_X = np.diag([-1.0, 1.0, 1.0])


def mirror_stereo(x: torch.Tensor, target: torch.Tensor, is_right: torch.Tensor):
    """Exact left<->right hand mirror: views share one virtual camera, so flip x and swap the views."""
    return (x.flip(-1).flip(-3), target * target.new_tensor([-1.0, 1.0, 1.0]), 1.0 - is_right)


class StereoIRDataset(Dataset):
    """
    PyTorch Dataset outputting:
      - stereo_tensor: (2, 128, 128) perspective crops, float32 in [-1, 1]
      - geo: (GEO_DIM,) hand distance / angular size features
      - joints_3d: (21, 3) mm, wrist-relative, in the virtual camera frame
      - leap_to_virtual: (3, 3); LeapC joints = joints_3d @ leap_to_virtual
      - is_right: () float, 1.0 for an Ultraleap-labelled right hand

    Crop geometry comes from the projected-joint boxes (jittered and rolled
    during training).
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        val_ratio: float = 0.2,
        target_size: int = 128,
        min_confidence: float = 0.0,
        max_roll_deg: float = 45.0,
        mirror_prob: float = 0.0,
    ):
        self.split = split.lower()
        self.mirror_prob = mirror_prob if self.split == "train" else 0.0
        self.preprocessor = StereoPerspectivePreprocessor(
            target_size=target_size,
            is_training=(self.split == "train"),
            max_roll_deg=max_roll_deg,
        )
        self.cached_data, self.meta = load_split_frames(dataset_dir, self.split, val_ratio, min_confidence,
                                                        with_meta=True)

    def __len__(self) -> int:
        return len(self.cached_data)

    def curl_balanced_weights(self) -> np.ndarray:
        """Per-frame sampling weights ~ 1/sqrt(frequency) of the grab-strength band
        (open < 0.2 <= half-curled < 0.8 <= fist), so rare curled poses are seen
        more often without repeating the few hundred half-curled frames too much."""
        grab = np.array([m["grab"] for m in self.meta])
        band = np.digitize(grab, [0.2, 0.8])
        freq = np.bincount(band, minlength=3) / len(band)
        return 1.0 / np.sqrt(freq[band])

    def __getitem__(self, idx: int):
        img_l, img_r, joints, bbox_l, bbox_r, luts = self.cached_data[idx]
        x, geo, geom = self.preprocessor(img_l, img_r, luts, bbox_l, bbox_r)
        rot = geom.leap_to_virtual
        target = torch.from_numpy((joints - joints[0:1]) @ rot.T).float()
        is_right = torch.tensor(self.meta[idx]["is_right"], dtype=torch.float32)
        if self.mirror_prob and random.random() < self.mirror_prob:
            x, target, is_right = mirror_stereo(x, target, is_right)
            rot = _MIRROR_X @ rot
        return x, geo, target, torch.from_numpy(rot).float(), is_right
