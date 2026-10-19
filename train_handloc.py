"""
train_handloc.py — Train HandLocNet (hand localisation in raw 3Di IR views).

Targets are the Ultraleap joints projected into each view, so no manual
labelling is needed. Uses the same session split as train_fist.py.

Usage:
  python train_handloc.py --dataset dataset_ir --epochs 30
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from fist_dataset import load_split_frames
from hand_locator import HandLocNet, locator_loss, make_targets, prepare_view, decode


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1)


class HandLocDataset(Dataset):
    """Every left and right view becomes one (image, heatmap, size) sample."""

    def __init__(self, frames, is_training: bool):
        self.views = []
        for img_l, img_r, _, bbox_l, bbox_r, _ in frames:
            self.views.append((img_l, bbox_l))
            self.views.append((img_r, bbox_r))
        self.is_training = is_training

    def __len__(self):
        return len(self.views)

    def __getitem__(self, idx):
        img, bbox = self.views[idx]
        if self.is_training:
            # Photometric jitter only: geometry must stay aligned with the box
            gain = random.uniform(0.6, 1.5)
            img = np.clip(img.astype(np.float32) * gain + random.uniform(-10, 10), 0, 255).astype(np.uint8)
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                x1, y1, x2, y2 = bbox
                w = img.shape[1]
                bbox = (w - x2, y1, w - x1, y2)
        heat, size = make_targets(bbox, img.shape[1])
        return prepare_view(img), torch.from_numpy(heat), torch.from_numpy(size), torch.tensor(bbox)


def main():
    parser = argparse.ArgumentParser(description="Train HandLocNet on stereo IR sessions")
    parser.add_argument("--dataset", type=str, default=str(config.DATASET_IR_DIR))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default=str(config.CHECKPOINTS_DIR / "handloc_best.pth"))
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    train_ds = HandLocDataset(load_split_frames(args.dataset, "train", 0.2), is_training=True)
    val_ds = HandLocDataset(load_split_frames(args.dataset, "val", 0.2), is_training=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = HandLocNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    print(f"[HANDLOC] {len(train_ds)} train views, {len(val_ds)} val views on {device}")
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, heat, size, _ in train_dl:
            x, heat, size = x.to(device), heat.to(device), size.to(device)
            loss = locator_loss(model(x), heat, size)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        sched.step()

        model.eval()
        ious, misses = [], 0
        with torch.no_grad():
            for x, _, _, gt in val_dl:
                out = model(x.to(device)).cpu()
                for o, g in zip(out, gt):
                    pred, _ = decode(o, (384, 384))
                    misses += pred is None
                    ious.append(box_iou(pred, tuple(g.tolist())))
        miou = float(np.mean(ious))
        mark = ""
        if miou > best_iou:
            best_iou = miou
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_mean_iou": miou,
                        "input_format": config.HANDLOC_INPUT_FORMAT}, args.out)
            mark = "[BEST]"
        print(f"Ep {epoch:02d}/{args.epochs} | loss {tr_loss / max(1, len(train_dl)):.3f} | "
              f"val mIoU {miou:.3f} | IoU>0.5 {np.mean(np.array(ious) > 0.5):.2%} | misses {misses} {mark}")

    print(f"[HANDLOC] Best val mIoU {best_iou:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
