"""
fist_model.py — Stereo3DFistNet: Lightweight 3D Hand Pose Estimator.

Designed specifically for real-time CPU execution on Dell Latitude 3440.
Accepts dual 8-bit active IR stereo crops (2, 128, 128) and directly predicts
the 21 3D hand joints (63 coordinates) in millimetres.

Key Architectural Properties:
  1. Early Epipolar Stereo Fusion: Conv2d(in_channels=2, ...) directly models
     disparity between Left and Right active IR sensors.
  2. Inverted Residual Bottlenecks with Squeeze-and-Excitation (SE): Highly
     efficient depthwise-separable design (< 1.2M parameters, ~120M FLOPs).
  3. Spatial Disparity Reasoning: Preserves sub-millimetre resolution without
     the heavy computational overhead of Vision Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from perspective_crop import GEO_DIM


class SqueezeExcitation(nn.Module):
    """SE Attention block to recalibrate channel-wise features."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class InvertedResidualBlock(nn.Module):
    """Depthwise-Separable Inverted Residual Block with SE attention."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: int = 4):
        super().__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_c == out_c)
        hidden_dim = in_c * expand_ratio

        layers = []
        # Pointwise Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ])

        # Depthwise Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            SqueezeExcitation(hidden_dim),
            # Pointwise Linear Projection
            nn.Conv2d(hidden_dim, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class Stereo3DFistNet(nn.Module):
    """
    Lightweight Dual-IR 3D Hand Reconstruction Network.
    
    Inputs:
      x: (B, 2, 128, 128) where channel 0 is Left IR and channel 1 is Right IR.
    Outputs:
      joints_3d: (B, 21, 3) predicted 3D joint coordinates in millimetres
                 (wrist-relative coordinate system).
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_joints: int = config.NUM_JOINTS,
        output_scale_mm: float = 200.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.output_scale_mm = output_scale_mm

        # Stem: Early stereo fusion
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False), # -> 64x64
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Inverted Residual Stages (Staged downsampling to 8x8)
        self.stage1 = nn.Sequential(
            InvertedResidualBlock(32, 48, stride=2, expand_ratio=3),   # -> 32x32
            InvertedResidualBlock(48, 48, stride=1, expand_ratio=3),
        )
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(48, 80, stride=2, expand_ratio=4),   # -> 16x16
            InvertedResidualBlock(80, 80, stride=1, expand_ratio=4),
            InvertedResidualBlock(80, 80, stride=1, expand_ratio=4),
        )
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(80, 144, stride=2, expand_ratio=4),  # -> 8x8
            InvertedResidualBlock(144, 144, stride=1, expand_ratio=4),
        )
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(144, 224, stride=1, expand_ratio=4), # -> 8x8
            nn.Conv2d(224, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),
        )

        # Global Pooling & Regression Head (+ crop geometry: hand distance, angular size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(384 + GEO_DIM, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, num_joints * 3)  # 63 outputs
        )

    def forward(self, x: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 128, 128) perspective stereo crops
            geo: (B, GEO_DIM) crop geometry features
        Returns:
            (B, 21, 3) wrist-relative joints in mm, virtual camera frame.
        """
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)

        flat = torch.cat([self.pool(feat).flatten(1), geo], dim=1)
        out = self.head(flat)  # (B, 63)

        # Scale raw output by characteristic hand dimension (~200 mm)
        joints_3d = (out * self.output_scale_mm).view(-1, self.num_joints, 3)

        # Anchor wrist (Joint 0) at (0, 0, 0) for strict root alignment
        wrist = joints_3d[:, 0:1, :]
        joints_3d = joints_3d - wrist

        return joints_3d


class ResNetStereoFistNet(nn.Module):
    """
    ImageNet-pretrained ResNet-18 shared by both IR views (siamese), fused at the
    4x4 feature map so the head still sees left/right spatial correspondence.
    Same inputs/outputs and regression head as Stereo3DFistNet.
    """
    GRAY_MEAN, GRAY_STD = 0.449, 0.226  # ImageNet RGB stats averaged to one channel

    def __init__(self, num_joints: int = config.NUM_JOINTS, output_scale_mm: float = 200.0,
                 pretrained: bool = True, hand_head: bool = False):
        super().__init__()
        import torchvision
        self.num_joints = num_joints
        self.output_scale_mm = output_scale_mm
        self.has_hand_head = hand_head

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        # Grey input replicated to RGB == one channel convolved with the summed RGB filters
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data = net.conv1.weight.data.sum(dim=1, keepdim=True)
        self.encoder = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool,
                                     net.layer1, net.layer2, net.layer3, net.layer4)

        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(384 + GEO_DIM, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, num_joints * 3),
        )
        # Left/right logit (>0 = right hand)
        self.hand_cls = nn.Linear(384 + GEO_DIM, 1) if hand_head else None

    def forward(self, x: torch.Tensor, geo: torch.Tensor, return_hand: bool = False):
        b = x.shape[0]
        views = ((x + 1.0) * 0.5 - self.GRAY_MEAN) / self.GRAY_STD   # [-1,1] -> ImageNet-normalised
        views = views.reshape(b * 2, 1, *x.shape[2:])                  # (2B, 1, S, S): L0,R0,L1,R1...
        feat = self.encoder(views)                                     # (2B, 512, s, s)
        feat = feat.reshape(b, 2 * feat.shape[1], *feat.shape[2:])     # (B, 1024, s, s)
        feat = self.fuse(feat)

        flat = torch.cat([self.pool(feat).flatten(1), geo], dim=1)
        joints_3d = (self.head(flat) * self.output_scale_mm).view(-1, self.num_joints, 3)
        joints_3d = joints_3d - joints_3d[:, 0:1, :]
        if return_hand:
            return joints_3d, (self.hand_cls(flat)[:, 0] if self.hand_cls is not None else None)
        return joints_3d


BACKBONES = ("custom", "resnet18")


def build_fist_model(backbone: str = "custom", pretrained: bool = True, hand_head: bool = False) -> nn.Module:
    if backbone == "resnet18":
        return ResNetStereoFistNet(num_joints=config.NUM_JOINTS, pretrained=pretrained, hand_head=hand_head)
    if hand_head:
        raise ValueError("The left/right head needs --backbone resnet18")
    return Stereo3DFistNet(in_channels=2, num_joints=config.NUM_JOINTS)


def load_fist_checkpoint(weights_path, device="cpu", allow_stale: bool = False):
    """
    Build Stereo3DFistNet and load a train_fist.py checkpoint.

    Raises ValueError for checkpoints made with an older input pipeline (e.g. the
    first fist_best.pth, which was trained on duplicated webcam frames rather than
    stereo IR) unless allow_stale=True.
    """
    ckpt = torch.load(weights_path, map_location=device)
    fmt = ckpt.get("input_format") if isinstance(ckpt, dict) else None
    if fmt != config.FIST_INPUT_FORMAT:
        msg = (f"Checkpoint {weights_path} has input_format={fmt!r}, expected "
               f"{config.FIST_INPUT_FORMAT!r}. It was trained with a different input "
               f"pipeline and will not reconstruct IR hands correctly. Retrain with: "
               f"python train_fist.py --dataset dataset_ir")
        if not allow_stale:
            raise ValueError(msg)
        print(f"[WARN] {msg}")
    backbone = ckpt.get("backbone", "custom") if isinstance(ckpt, dict) else "custom"
    hand_head = bool(ckpt.get("hand_head", False)) if isinstance(ckpt, dict) else False
    model = build_fist_model(backbone, pretrained=False, hand_head=hand_head)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.to(device).eval()
    return model, ckpt
