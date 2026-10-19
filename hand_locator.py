"""
hand_locator.py — Learned hand localisation in a single raw 3Di IR view.

Stereo3DFistNet needs a hand-centred crop, and at test time that crop must
come from the IR image alone. In the front-facing 3Di view the hand is not
reliably the brightest object (the lit torso and desk are as bright), so a
threshold heuristic fails. HandLocNet is a tiny CenterNet-style detector:

  input : (1, 128, 128) full IR view, downsampled from 384x384, in [-1, 1]
  output: (2, 32, 32)   channel 0 = hand-centre heatmap logits,
                        channel 1 = box side / image width at each cell

Supervision comes for free: training boxes are the Ultraleap joints projected
into each view (ir_geometry.py).
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

LOC_INPUT = 128
LOC_STRIDE = 4
LOC_GRID = LOC_INPUT // LOC_STRIDE


def _conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class HandLocNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(_conv(1, 24, 2), _conv(24, 24))      # 64
        self.enc2 = nn.Sequential(_conv(24, 48, 2), _conv(48, 48))     # 32
        self.enc3 = nn.Sequential(_conv(48, 96, 2), _conv(96, 96))     # 16
        self.enc4 = nn.Sequential(_conv(96, 128, 2), _conv(128, 128))  # 8
        self.up3 = _conv(128 + 96, 96)
        self.up2 = _conv(96 + 48, 64)
        self.head = nn.Conv2d(64, 2, 1)
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():
            self.head.bias[0] = -4.0  # heatmap prior: background

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d3 = self.up3(torch.cat([F.interpolate(e4, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.up2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        return self.head(d2)  # (B, 2, 32, 32)


def prepare_view(img: np.ndarray) -> torch.Tensor:
    small = cv2.resize(img, (LOC_INPUT, LOC_INPUT), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(small.astype(np.float32) / 127.5 - 1.0)[None]  # (1, H, W)


def make_targets(bbox: Optional[Tuple[int, int, int, int]], img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian centre heatmap (G, G) and side-length target (G, G) for one box."""
    heat = np.zeros((LOC_GRID, LOC_GRID), np.float32)
    size = np.zeros((LOC_GRID, LOC_GRID), np.float32)
    if bbox is None:
        return heat, size
    x1, y1, x2, y2 = bbox
    s = LOC_GRID / img_size
    # Peak exactly 1.0 on the cell containing the centre, so every box yields
    # one positive for the focal loss (CenterNet convention)
    gx = min(LOC_GRID - 1, int((x1 + x2) / 2 * s))
    gy = min(LOC_GRID - 1, int((y1 + y2) / 2 * s))
    side = (x2 - x1) / img_size
    sigma = max(1.0, side * LOC_GRID / 6)
    ys, xs = np.mgrid[0:LOC_GRID, 0:LOC_GRID]
    heat[:] = np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * sigma ** 2))
    size[:] = side
    return heat, size


def locator_loss(out: torch.Tensor, heat: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced focal loss on the heatmap + L1 on size near the centre."""
    logits, pred_size = out[:, 0], out[:, 1]
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (heat > 0.99).float()
    neg_w = (1 - heat) ** 4
    focal = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * neg_w * p ** 2 * torch.log(1 - p))
    n_pos = pos.sum().clamp(min=1.0)
    w = (heat > 0.5).float()
    size_l1 = (w * (pred_size - size).abs()).sum() / w.sum().clamp(min=1.0)
    return focal.sum() / n_pos + 5.0 * size_l1


def decode(out: torch.Tensor, img_shape, min_score: float = 0.3):
    """(2, G, G) network output -> (square bbox in pixels, score) or (None, score)."""
    heat = torch.sigmoid(out[0])
    score, idx = heat.flatten().max(0)
    score = float(score)
    if score < min_score:
        return None, score
    gy, gx = divmod(int(idx), LOC_GRID)
    h, w = img_shape[:2]
    cx = (gx + 0.5) / LOC_GRID * w
    cy = (gy + 0.5) / LOC_GRID * h
    side = int(np.clip(float(out[1, gy, gx]) * w, 32, min(h, w)))
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return (x1, y1, x1 + side, y1 + side), score


def measure_box_errors(locator: "HandLocator", frames) -> np.ndarray:
    """Locator vs projected-joint boxes on load_split_frames() frames ->
    (M, 3) rows of (centre dx / side, centre dy / side, side ratio), one per view."""
    errs = []
    for img_l, img_r, _, bbox_l, bbox_r, _ in frames:
        pred_l, pred_r, _ = locator.locate_pair(img_l, img_r)
        for p, g in ((pred_l, bbox_l), (pred_r, bbox_r)):
            if p is None:
                continue
            side = g[2] - g[0]
            errs.append(((p[0] + p[2] - g[0] - g[2]) / 2 / side,
                         (p[1] + p[3] - g[1] - g[3]) / 2 / side,
                         (p[2] - p[0]) / side))
    return np.array(errs, dtype=np.float32).reshape(-1, 3)


class HandLocator:
    """Inference wrapper: raw IR view -> hand box."""

    def __init__(self, weights_path, device="cpu", min_score: float = 0.3):
        ckpt = torch.load(weights_path, map_location=device)
        if ckpt.get("input_format") != config.HANDLOC_INPUT_FORMAT:
            raise ValueError(f"{weights_path} is not a {config.HANDLOC_INPUT_FORMAT} checkpoint; "
                             f"retrain with: python train_handloc.py")
        self.model = HandLocNet().to(device).eval()
        self.model.load_state_dict(ckpt["model_state"])
        self.device = device
        self.min_score = min_score

    @torch.no_grad()
    def locate_pair(self, left: np.ndarray, right: np.ndarray):
        x = torch.stack([prepare_view(left), prepare_view(right)]).to(self.device)
        out = self.model(x).cpu()
        (bl, sl), (br, sr) = decode(out[0], left.shape, self.min_score), decode(out[1], right.shape, self.min_score)
        return bl, br, (sl, sr)
