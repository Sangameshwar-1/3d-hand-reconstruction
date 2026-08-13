"""
inference.py — Stage 4a: Baseline Inference (MediaPipe / HaMeR).

Runs a baseline monocular hand pose estimation model on the captured RGB frames.
Outputs predicted 3D joints for evaluation.

Usage:
    python inference.py --dataset dataset/ --model mediapipe
"""

import argparse
import sys
import json
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import torch
import torchvision.transforms as T

import config
from dataset_utils import load_dataset_pairs
from calibrate_intrinsics import load_intrinsics
from finetune import FineTunedHandPoseModel


class HandPoseEstimator(ABC):
    """Abstract base class for hand pose estimators."""
    
    def __init__(self, K: np.ndarray = None):
        self.K = K  # Camera matrix (needed to unproject normalized 2D to 3D)

    @abstractmethod
    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Predict 3D hand joints from an RGB image.
        Returns:
            (21, 3) numpy array of joint coordinates in millimetres (Camera space).
            Returns None if no hand is detected.
        """
        pass


class MediaPipeEstimator(HandPoseEstimator):
    """
    Wrapper for MediaPipe Hands.
    MediaPipe outputs normalized [0,1] coordinates. We use a heuristic and
    camera intrinsics to scale this to absolute millimetres.
    """
    
    def __init__(self, K: np.ndarray, static_image_mode=True):
        super().__init__(K)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Focal lengths
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        # MediaPipe heuristic: the 'z' value is roughly proportional to the 
        # wrist width. To get absolute depth, we need an anchor.
        # Here we use a generic anatomical heuristic: 
        # Average distance from Wrist to Middle MCP is ~100mm.
        self.REF_BONE_MM = 100.0

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        h, w = rgb_image.shape[:2]
        
        # MediaPipe expects RGB
        rgb_input = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_input)

        if not results.multi_hand_landmarks:
            return None

        # Take the first hand
        landmarks = results.multi_hand_landmarks[0].landmark

        # Extract normalized 3D
        joints_norm = np.zeros((config.NUM_JOINTS, 3))
        for i, lm in enumerate(landmarks):
            joints_norm[i] = [lm.x, lm.y, lm.z]

        # Denormalize X and Y to pixels
        px = joints_norm[:, 0] * w
        py = joints_norm[:, 1] * h
        
        # MediaPipe Z is roughly scaled so that Z-spread == X-spread.
        # But we want absolute mm depth.
        # Strategy:
        # 1. Find 2D pixel distance between Wrist (0) and Middle MCP (9)
        px_dist = np.sqrt((px[0] - px[9])**2 + (py[0] - py[9])**2)
        
        # 2. Use similar triangles to estimate absolute Z depth of the hand:
        #    Z = (focal_length * real_bone_len) / pixel_bone_len
        #    Using fx for approximation.
        focal = (self.fx + self.fy) / 2.0
        abs_depth = (focal * self.REF_BONE_MM) / (px_dist + 1e-5)

        # 3. Unproject 2D pixels back to 3D rays
        X = (px - self.cx) / self.fx
        Y = (py - self.cy) / self.fy

        # 4. Scale rays by depth to get mm.
        # Add MP's relative Z variation back in (scaled appropriately)
        # MP Z range is roughly proportional to the hand width in pixels.
        z_scale_factor = abs_depth / focal  # mm per pixel at that depth
        relative_z_mm = joints_norm[:, 2] * w * z_scale_factor

        Z = abs_depth + relative_z_mm
        X = X * Z
        Y = Y * Z

        # Stack to (21, 3)
        joints_3d = np.column_stack((X, Y, Z))
        
        # Alignment: MediaPipe Z is negative-towards-camera (right-handed with 
        # y-down). Our Leap extrinsics usually output Z positive-away.
        # So we leave Z as positive (away from camera).
        
        return joints_3d


class ViTEstimator(HandPoseEstimator):
    """
    Wrapper for our fine-tuned Vision Transformer.
    """
    def __init__(self, K: np.ndarray):
        super().__init__(K)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_path = config.CHECKPOINTS_DIR / "best_model.pth"
        if not model_path.exists():
            print(f"[ERROR] Fine-tuned model weights not found at {model_path}")
            sys.exit(1)
            
        self.model = FineTunedHandPoseModel(freeze_backbone=True)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Transform to tensor (C, H, W) and add batch dim (1, C, H, W)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # Reshape flat output to (21, 3)
        joints_3d = output.cpu().numpy().reshape(config.NUM_JOINTS, 3)
        return joints_3d


def run_inference_on_dataset(dataset_dir: Path, estimator: HandPoseEstimator,
                             model_name: str):
    """Run the estimator on all RGB frames and save predictions."""
    pairs = load_dataset_pairs(dataset_dir)
    if not pairs:
        print("[ERROR] No pairs found in dataset.")
        return

    out_dir = dataset_dir / f"predictions_{model_name}"
    out_dir.mkdir(exist_ok=True)
    
    print(f"\n[INFERENCE] Running {model_name} on {len(pairs)} frames...")
    
    success_count = 0
    for rgb_path, _ in tqdm(pairs, desc="Inference"):
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            continue
            
        joints_pred = estimator.predict(frame)
        
        if joints_pred is not None:
            # Save prediction
            # Convert to dict format for consistency
            joints_dict = {}
            for i, name in config.JOINT_NAMES.items():
                joints_dict[f"{i}_{name}"] = {
                    "x": float(joints_pred[i, 0]),
                    "y": float(joints_pred[i, 1]),
                    "z": float(joints_pred[i, 2]),
                }
                
            out_file = out_dir / (rgb_path.stem + ".json")
            with open(out_file, "w") as f:
                json.dump({"joints": joints_dict}, f, indent=2)
            success_count += 1
            
    print(f"\n[INFERENCE] Complete. Predictions saved to {out_dir}")
    print(f"  Successfully detected hands in {success_count}/{len(pairs)} frames.")


def main():
    parser = argparse.ArgumentParser(description="Stage 4a: Baseline Inference")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--model", type=str, default="mediapipe",
                        choices=["mediapipe", "hamer", "finetuned"],
                        help="Baseline model to use")
    parser.add_argument("--intrinsics", type=str,
                        default=str(config.CALIBRATION_DIR / "camera_intrinsics.json"))
    args = parser.parse_args()

    # Load intrinsics
    intrinsics_path = Path(args.intrinsics)
    if not intrinsics_path.exists():
        print("[ERROR] Intrinsics required to scale 2D to 3D.")
        sys.exit(1)
    K, _ = load_intrinsics(intrinsics_path)

    # Init estimator
    if args.model == "mediapipe":
        estimator = MediaPipeEstimator(K)
    elif args.model == "finetuned":
        estimator = ViTEstimator(K)
    else:
        print(f"[ERROR] Model wrapper for '{args.model}' not implemented.")
        sys.exit(1)

    dataset_dir = Path(args.dataset)
    run_inference_on_dataset(dataset_dir, estimator, args.model)


if __name__ == "__main__":
    main()
