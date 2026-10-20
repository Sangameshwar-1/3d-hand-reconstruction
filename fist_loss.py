"""
fist_loss.py — Physics-Informed Kinematic Loss for Self-Occluded Palmar Poses.

Addresses the two primary physical challenges in front-facing palmar views:
  1. Active IR Saturation: Uses Smooth-L1 (Huber) loss instead of standard MSE
     to prevent gradient explosion when active LEDs saturate palmar pixels.
  2. Finger Self-Occlusion:
     - Bone-Length Invariance (L_bone): When fingertips curl inward behind
       proximal joints, bone lengths remain strictly constant.
     - Anatomical Flexion Hinge Limit (L_angle): Prevents non-physical
       accordion folding and hyperextension of finger hinges.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
FINGER_NAMES = ("T", "I", "M", "R", "P")
_FINGER_CHAINS = torch.tensor([(0, *f) for f in FINGERS])


def finger_curl(j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-finger (flexion, signed curl), each (B, 5) for joints (B, 21, 3).

    flexion: mean (1 - cos) over the knuckle, middle and end joints; 0 = straight, 1 = 90 deg at each.
    signed curl: tip-minus-knuckle offset along the palm normal / finger length. Its sign is the
    direction the finger folds, so a left fist and a right fist curl opposite ways.
    """
    n = F.normalize(torch.cross(j[:, 9] - j[:, 0], j[:, 5] - j[:, 17], dim=-1), dim=-1, eps=1e-6)
    pts = j[:, _FINGER_CHAINS.to(j.device)]  # (B, 5, 5, 3): wrist, then the finger's 4 joints
    bones = pts[:, :, 1:] - pts[:, :, :-1]
    u = F.normalize(bones, dim=-1, eps=1e-6)
    flex = (1.0 - (u[:, :, 1:] * u[:, :, :-1]).sum(-1)).mean(-1)
    length = bones[:, :, 1:].norm(dim=-1).sum(-1).clamp_min(1.0)
    signed = ((pts[:, :, 4] - pts[:, :, 1]) * n[:, None]).sum(-1) / length
    return flex, signed


class PhysicsInformedFistLoss(nn.Module):
    """
    Combined Kinematic & Anisotropic Loss for 3DFist.

    L = L_coord + lambda_bone * L_bone + lambda_angle * L_angle + lambda_curl * L_curl
    """

    # Inter-phalangeal hinge triplets (Base, Hinge, Tip)
    HINGE_TRIPLETS = [
        # Thumb
        (1, 2, 3), (2, 3, 4),
        # Index
        (5, 6, 7), (6, 7, 8),
        # Middle
        (9, 10, 11), (10, 11, 12),
        # Ring
        (13, 14, 15), (14, 15, 16),
        # Pinky
        (17, 18, 19), (18, 19, 20),
    ]

    # cos(60 deg): hinge angles tighter than this are anatomically impossible
    MAX_FOLD_COS = 0.5

    def __init__(
        self,
        lambda_xy: float = 1.0,
        lambda_z: float = 3.0,
        lambda_bone: float = 1.5,
        lambda_angle: float = 1.0,
        huber_beta: float = 2.0,  # mm transition threshold for SmoothL1
        lambda_curl: float = 0.0,
    ):
        super().__init__()
        self.lambda_xy = lambda_xy
        self.lambda_z = lambda_z
        self.lambda_bone = lambda_bone
        self.lambda_angle = lambda_angle
        self.lambda_curl = lambda_curl

        self.smooth_l1 = nn.SmoothL1Loss(beta=huber_beta, reduction="none")

        # Bone index pairs from config
        self.bone_starts = [pair[0] for pair in config.SKELETON_CONNECTIONS]
        self.bone_ends = [pair[1] for pair in config.SKELETON_CONNECTIONS]

    def compute_bone_loss(
        self,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforces that predicted bone lengths match ground-truth biological lengths:
        L_bone = mean( | ||p_start - p_end|| - ||gt_start - gt_end|| | )
        """
        pred_s = pred_joints[:, self.bone_starts, :]
        pred_e = pred_joints[:, self.bone_ends, :]
        pred_len = torch.norm(pred_s - pred_e + 1e-8, dim=-1)  # (B, 20)

        gt_s = gt_joints[:, self.bone_starts, :]
        gt_e = gt_joints[:, self.bone_ends, :]
        gt_len = torch.norm(gt_s - gt_e + 1e-8, dim=-1)        # (B, 20)

        # L1 discrepancy between predicted and GT bone lengths
        bone_loss = F.l1_loss(pred_len, gt_len)
        return bone_loss

    def compute_angular_loss(self, pred_joints: torch.Tensor) -> torch.Tensor:
        """
        Penalizes non-physical accordion folding (hyperextension).
        For hinge (a, b, c):
          v1 = j_a - j_b (toward base)
          v2 = j_c - j_b (toward tip)
        cos(theta) = v1 . v2
        A straight finger gives theta = 180 deg (cos = -1); flexion shrinks theta.
        A tight fist bends PIP/DIP joints by ~90-110 deg (theta ~70-90 deg,
        cos up to ~0.35), which must NOT be penalised. Only folding past the
        anatomical limit (theta < 60 deg, cos > 0.5) is treated as a violation.
        """
        loss_angle = pred_joints.new_zeros(1)

        for a, b, c in self.HINGE_TRIPLETS:
            v1 = pred_joints[:, a, :] - pred_joints[:, b, :]
            v2 = pred_joints[:, c, :] - pred_joints[:, b, :]

            v1_u = F.normalize(v1, p=2, dim=-1, eps=1e-8)
            v2_u = F.normalize(v2, p=2, dim=-1, eps=1e-8)

            cos_theta = torch.sum(v1_u * v2_u, dim=-1)  # (B,)
            # Penalize folding beyond physical threshold
            violation = torch.relu(cos_theta - self.MAX_FOLD_COS)
            loss_angle = loss_angle + violation.mean()

        return loss_angle / len(self.HINGE_TRIPLETS)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: (B, 21, 3) predicted joints in mm.
            gt: (B, 21, 3) ground-truth joints in mm.
        Returns:
            total_loss: scalar tensor.
            loss_dict: dictionary of individual loss components.
        """
        pred_3d = pred.view(-1, config.NUM_JOINTS, 3)
        gt_3d = gt.view(-1, config.NUM_JOINTS, 3)

        # 1. Anisotropic Spatial Coordinate Loss
        diff = self.smooth_l1(pred_3d, gt_3d)  # (B, 21, 3)
        weights = torch.tensor(
            [self.lambda_xy, self.lambda_xy, self.lambda_z],
            dtype=diff.dtype, device=diff.device
        ).view(1, 1, 3)
        coord_loss = (diff * weights).mean()

        # 2. Bone Length Invariance Loss
        bone_loss = self.compute_bone_loss(pred_3d, gt_3d)

        # 3. Anatomical Hinge Flexion Limit Loss
        angle_loss = self.compute_angular_loss(pred_3d)

        # 4. Per-finger curl: how far, and which way, each finger folds
        pf, ps = finger_curl(pred_3d)
        gf, gs = finger_curl(gt_3d)
        curl_loss = F.l1_loss(pf, gf) + F.l1_loss(ps, gs)

        # Combined Total
        total_loss = (coord_loss + (self.lambda_bone * bone_loss) + (self.lambda_angle * angle_loss)
                      + self.lambda_curl * curl_loss)

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_coord": coord_loss.item(),
            "loss_bone": bone_loss.item(),
            "loss_angle": angle_loss.item(),
            "loss_curl": curl_loss.item(),
        }

        return total_loss, loss_dict
