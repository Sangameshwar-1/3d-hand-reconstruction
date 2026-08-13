"""
hand_dataset.py — Stage 5a: PyTorch Dataset for Fine-tuning.

Creates a PyTorch Dataset that loads synchronized RGB frames and root-aligned
3D joints for training a hand pose regression model.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import numpy as np

import config
from dataset_utils import load_dataset_pairs, load_joint_json, joints_dict_to_array
from evaluate import align_to_root


class HandPoseDataset(Dataset):
    """
    PyTorch Dataset for RGB to 3D Hand Pose.
    """
    
    def __init__(self, dataset_dir: str, split: str = "train", 
                 transform=None, image_size: int = config.IMAGE_SIZE):
        """
        Args:
            dataset_dir: Path to the root dataset directory.
            split: "train" or "val".
            transform: Custom torchvision transforms. If None, default is used.
            image_size: Target square size for the model.
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        
        # Load all valid pairs
        all_pairs = load_dataset_pairs(self.dataset_dir)
        
        # Simple split based on config
        num_train = int(len(all_pairs) * config.TRAIN_SPLIT)
        if split == "train":
            self.pairs = all_pairs[:num_train]
        else:
            self.pairs = all_pairs[num_train:]
            
        print(f"[DATASET] Loaded {len(self.pairs)} pairs for {split} split.")
        
        # Default transforms for ViT models (ImageNet norms)
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((self.image_size, self.image_size), antialias=True),
                T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_path, leap_path = self.pairs[idx]
        
        # 1. Load and process image
        img_bgr = cv2.imread(str(rgb_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)
        
        # 2. Load and process joints
        joint_data = load_joint_json(leap_path)
        joints_arr = joints_dict_to_array(joint_data["joints"])
        
        # Root align the joints before training!
        joints_aligned = align_to_root(joints_arr)
        
        # Flatten (21, 3) -> (63,)
        joints_flat = joints_aligned.flatten()
        joints_tensor = torch.tensor(joints_flat, dtype=torch.float32)
        
        return img_tensor, joints_tensor
