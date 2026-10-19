"""
train_fist.py — Training Pipeline for 3DFist.

Trains Stereo3DFistNet with PhysicsInformedFistLoss on paired dual-IR frames
and Ultraleap 3D ground-truth coordinates.

Evaluates validation MPJPE and PA-MPJPE after every epoch, saving the champion
checkpoint based on lowest validation MPJPE.

Usage:
  python train_fist.py --dataset dataset_ir --epochs 40 --batch-size 8 --lr 1e-3
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

import config
from fist_model import BACKBONES, build_fist_model
from fist_loss import PhysicsInformedFistLoss
from fist_dataset import StereoIRDataset, mirror_stereo
from fist_metrics import compute_mpjpe, compute_pa_mpjpe
from hand_locator import HandLocator, measure_box_errors


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedFistLoss,
    device: torch.device,
    frozen_encoder: bool = False,
    hand_weight: float = 0.0,
) -> dict:
    model.train()
    if frozen_encoder:
        model.encoder.eval()  # keep pretrained BatchNorm statistics while frozen
    use_hand = getattr(model, "has_hand_head", False)
    total_loss = 0.0
    coord_loss = 0.0
    bone_loss = 0.0
    angle_loss = 0.0
    hand_loss = 0.0
    num_batches = len(dataloader)

    for stereo_imgs, geo, gt_joints, _, is_right in dataloader:
        stereo_imgs = stereo_imgs.to(device)
        geo = geo.to(device)
        gt_joints = gt_joints.to(device)

        optimizer.zero_grad()
        if use_hand:
            pred_joints, hand_logit = model(stereo_imgs, geo, return_hand=True)
        else:
            pred_joints = model(stereo_imgs, geo)

        loss, l_dict = criterion(pred_joints, gt_joints)
        if use_hand:
            l_hand = F.binary_cross_entropy_with_logits(hand_logit, is_right.to(device))
            loss = loss + hand_weight * l_hand
            hand_loss += l_hand.item()
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += l_dict["loss_total"]
        coord_loss += l_dict["loss_coord"]
        bone_loss += l_dict["loss_bone"]
        angle_loss += l_dict["loss_angle"]

    return {
        "train_loss": total_loss / num_batches,
        "train_coord": coord_loss / num_batches,
        "train_bone": bone_loss / num_batches,
        "train_angle": angle_loss / num_batches,
        "train_hand": hand_loss / num_batches,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: PhysicsInformedFistLoss,
    device: torch.device,
) -> dict:
    model.eval()
    use_hand = getattr(model, "has_hand_head", False)
    val_loss = 0.0
    all_preds = []
    all_gts = []
    hand_correct = 0
    hand_correct_mirror = 0

    with torch.no_grad():
        for stereo_imgs, geo, gt_joints, _, is_right in dataloader:
            stereo_imgs = stereo_imgs.to(device)
            gt_joints = gt_joints.to(device)

            if use_hand:
                pred_joints, hand_logit = model(stereo_imgs, geo.to(device), return_hand=True)
                hand_correct += int(((hand_logit.cpu() > 0).float() == is_right).sum())
                # Val sessions are nearly all left hands; their mirror images test right hands
                m_imgs, _, m_right = mirror_stereo(stereo_imgs, gt_joints, is_right)
                _, m_logit = model(m_imgs, geo.to(device), return_hand=True)
                hand_correct_mirror += int(((m_logit.cpu() > 0).float() == m_right).sum())
            else:
                pred_joints = model(stereo_imgs, geo.to(device))
            loss, _ = criterion(pred_joints, gt_joints)
            val_loss += loss.item()

            all_preds.append(pred_joints.cpu().numpy())
            all_gts.append(gt_joints.cpu().numpy())

    num_batches = max(1, len(dataloader))
    val_loss /= num_batches

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 21, 3)
    all_gts = np.concatenate(all_gts, axis=0)      # (N, 21, 3)

    mpjpe, _, axis_err = compute_mpjpe(all_gts, all_preds)
    pa_mpjpe, _ = compute_pa_mpjpe(all_gts, all_preds)

    return {
        "val_loss": val_loss,
        "val_mpjpe_mm": mpjpe,
        "val_pa_mpjpe_mm": pa_mpjpe,
        "val_z_error_mm": float(axis_err[2]),
        "val_x_error_mm": float(axis_err[0]),
        "val_y_error_mm": float(axis_err[1]),
        "val_hand_acc": hand_correct / len(all_preds) if use_hand else None,
        "val_hand_acc_mirror": hand_correct_mirror / len(all_preds) if use_hand else None,
    }


def main():
    parser = argparse.ArgumentParser(description="3DFist Training Pipeline")
    parser.add_argument("--dataset", type=str, default="dataset_ir",
                        help="Dataset directory containing left/, right/, leap/")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Drop frames whose Ultraleap hand confidence is below this")
    parser.add_argument("--handloc", type=str, default=None,
                        help="HandLocNet weights: replay its measured box errors as the training "
                             "crop jitter (instead of uniform jitter)")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker processes for crop rendering (use 4-8 on a GPU server)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Continue from a checkpoint (fist_last.pth, or fist_best.pth) at its next epoch")
    parser.add_argument("--backbone", type=str, default="custom", choices=BACKBONES,
                        help="custom = Stereo3DFistNet from scratch; resnet18 = ImageNet-pretrained "
                             "ResNet-18 shared by both views")
    parser.add_argument("--freeze-epochs", type=int, default=3,
                        help="Pretrained backbones: train only fusion+head for this many epochs")
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1,
                        help="Pretrained backbones: encoder learning rate = lr * this")
    parser.add_argument("--run-name", type=str, default="fist",
                        help="Prefix for checkpoints (<run>_best.pth, <run>_last.pth) and the CSV log")
    parser.add_argument("--max-roll", type=float, default=45.0,
                        help="Training in-plane rotation range (+/- deg) of the virtual crop camera; "
                             "180 covers every hand direction")
    parser.add_argument("--balance-curl", action="store_true",
                        help="Oversample half-curled and fist frames (grab strength bands)")
    parser.add_argument("--hand-head", action="store_true",
                        help="Also predict left/right hand (resnet18 backbone only)")
    parser.add_argument("--hand-weight", type=float, default=2.0,
                        help="Weight of the left/right BCE loss")
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror half the training samples (left hand <-> right hand)")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[3DFIST] Training on device: {device}")

    # Datasets and Loaders
    train_dataset = StereoIRDataset(args.dataset, split="train", val_ratio=0.2,
                                    min_confidence=args.min_confidence, max_roll_deg=args.max_roll,
                                    mirror_prob=0.5 if args.mirror else 0.0)
    val_dataset = StereoIRDataset(args.dataset, split="val", val_ratio=0.2,
                                  min_confidence=args.min_confidence)

    if args.handloc:
        # Measured on held-out sessions: on its own training sessions the locator is
        # near-perfect, which would understate the error seen at test time.
        errs = measure_box_errors(HandLocator(args.handloc, device), val_dataset.cached_data)
        # Gross misses (box on the wrong object) carry no hand to learn from
        errs = errs[np.linalg.norm(errs[:, :2], axis=1) < 0.5]
        train_dataset.preprocessor.box_errors = errs
        centre = np.linalg.norm(errs[:, :2], axis=1)
        print(f"[3DFIST] Crop jitter = HandLocNet errors ({len(errs)} views): centre offset "
              f"median {np.median(centre):.3f} / p90 {np.percentile(centre, 90):.3f} of box side, "
              f"side ratio median {np.median(errs[:, 2]):.3f}")

    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0,
                     pin_memory=device.type == "cuda")
    if args.balance_curl:
        sampler = WeightedRandomSampler(train_dataset.curl_balanced_weights(), len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # Model & Loss
    model = build_fist_model(args.backbone, hand_head=args.hand_head).to(device)
    has_encoder = hasattr(model, "encoder")
    criterion = PhysicsInformedFistLoss(
        lambda_xy=1.0,
        lambda_z=3.0,
        lambda_bone=1.5,
        lambda_angle=1.0,
    ).to(device)

    if has_encoder:
        enc_ids = {id(p) for p in model.encoder.parameters()}
        param_groups = [
            {"params": list(model.encoder.parameters()), "lr": args.lr * args.encoder_lr_scale,
             "lr_scale": args.encoder_lr_scale},
            {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.lr,
             "lr_scale": 1.0},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": args.lr, "lr_scale": 1.0}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    checkpoints_dir = config.CHECKPOINTS_DIR
    checkpoints_dir.mkdir(exist_ok=True)
    best_weights_path = checkpoints_dir / f"{args.run_name}_best.pth"
    last_weights_path = checkpoints_dir / f"{args.run_name}_last.pth"

    start_epoch = 1
    best_val_mpjpe = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get("input_format") != config.FIST_INPUT_FORMAT:
            print(f"[ERROR] {args.resume} has input_format={ckpt.get('input_format')!r}, "
                  f"expected {config.FIST_INPUT_FORMAT!r}")
            sys.exit(1)
        if ckpt.get("backbone", "custom") != args.backbone or bool(ckpt.get("hand_head", False)) != args.hand_head:
            print(f"[ERROR] {args.resume} uses backbone {ckpt.get('backbone', 'custom')!r} "
                  f"(hand head {bool(ckpt.get('hand_head', False))}), but --backbone {args.backbone!r} "
                  f"(hand head {args.hand_head}) was given")
            sys.exit(1)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for g in optimizer.param_groups:
            g["initial_lr"] = args.lr * g.get("lr_scale", 1.0)
        start_epoch = ckpt["epoch"] + 1
        best_val_mpjpe = ckpt.get("best_val_mpjpe_mm", ckpt.get("val_mpjpe_mm", float("inf")))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5, last_epoch=ckpt["epoch"] - 1)
        scheduler.step()
        print(f"[3DFIST] Resumed {args.resume} at epoch {start_epoch} "
              f"(best val MPJPE so far {best_val_mpjpe:.2f} mm, lr {scheduler.get_last_lr()[0]:.2e})")

    results_dir = config.RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    log_csv_path = results_dir / f"{args.run_name}_training_log.csv"

    # CSV Logger Header (appended to when resuming)
    fieldnames = [
        "epoch", "train_loss", "train_coord", "train_bone", "train_angle",
        "val_loss", "val_mpjpe_mm", "val_pa_mpjpe_mm", "val_z_error_mm", "lr"
    ]
    if args.hand_head:
        fieldnames += ["train_hand", "val_hand_acc", "val_hand_acc_mirror"]
    if not (args.resume and log_csv_path.exists()):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    print()
    print("=" * 72)
    print("  3DFist -- Physics-Informed Stereo IR Model Training")
    arch = ("ResNet-18 (ImageNet) shared across views, 4x4 feature fusion" if has_encoder
            else "Stereo3DFistNet (Inverted Residuals + SE, in_ch=2)")
    print(f"  Architecture: {arch}  [{sum(p.numel() for p in model.parameters()):,} params]")
    if has_encoder:
        print(f"  Encoder:      frozen for {args.freeze_epochs} epochs, then lr x {args.encoder_lr_scale}")
    print(f"  Augment:      roll +/-{args.max_roll:.0f} deg | curl-balanced sampling: {args.balance_curl} | "
          f"left/right head: {args.hand_head} | mirror: {args.mirror}")
    print(f"  Samples:      {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"  Input Shape:  (2, 128, 128) perspective stereo IR crops | Target: 21 Joints (mm, virtual camera frame)")
    print("=" * 72)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        frozen = has_encoder and epoch <= args.freeze_epochs
        if has_encoder:
            for p in model.encoder.parameters():
                p.requires_grad_(not frozen)
        train_res = train_one_epoch(model, train_loader, optimizer, criterion, device, frozen, args.hand_weight)
        val_res = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        cur_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Save champion checkpoint
        is_best = val_res["val_mpjpe_mm"] < best_val_mpjpe
        if is_best:
            best_val_mpjpe = val_res["val_mpjpe_mm"]
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_mpjpe_mm": val_res["val_mpjpe_mm"],
            "val_pa_mpjpe_mm": val_res["val_pa_mpjpe_mm"],
            "best_val_mpjpe_mm": best_val_mpjpe,
            "input_format": config.FIST_INPUT_FORMAT,
            "backbone": args.backbone,
            "hand_head": args.hand_head,
            "dataset": str(args.dataset),
        }
        torch.save(ckpt, last_weights_path)
        if is_best:
            torch.save(ckpt, best_weights_path)
            best_mark = "[BEST]"
        else:
            best_mark = ""

        # Print Epoch Summary
        print(
            f"Ep {epoch:02d}/{args.epochs:02d} | "
            f"Tr Loss: {train_res['train_loss']:.3f} | "
            f"Val Loss: {val_res['val_loss']:.3f} | "
            f"MPJPE: {val_res['val_mpjpe_mm']:6.2f} mm | "
            f"PA-MPJPE: {val_res['val_pa_mpjpe_mm']:6.2f} mm | "
            f"Z Err: {val_res['val_z_error_mm']:5.2f} mm | "
            + (f"L/R acc: {val_res['val_hand_acc'] * 100:5.1f}% / mirrored {val_res['val_hand_acc_mirror'] * 100:5.1f}% | "
               if args.hand_head else "")
            + f"{elapsed:4.1f}s {best_mark}"
        )

        # Log row to CSV
        row_extra = ({"train_hand": round(train_res["train_hand"], 4),
                      "val_hand_acc": round(val_res["val_hand_acc"], 4),
                      "val_hand_acc_mirror": round(val_res["val_hand_acc_mirror"], 4)} if args.hand_head else {})
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                **row_extra,
                "epoch": epoch,
                "train_loss": round(train_res["train_loss"], 4),
                "train_coord": round(train_res["train_coord"], 4),
                "train_bone": round(train_res["train_bone"], 4),
                "train_angle": round(train_res["train_angle"], 4),
                "val_loss": round(val_res["val_loss"], 4),
                "val_mpjpe_mm": round(val_res["val_mpjpe_mm"], 2),
                "val_pa_mpjpe_mm": round(val_res["val_pa_mpjpe_mm"], 2),
                "val_z_error_mm": round(val_res["val_z_error_mm"], 2),
                "lr": f"{cur_lr:.6f}",
            })

    total_time = time.time() - start_time
    print()
    print("=" * 72)
    print(f"  Training Complete in {total_time:.1f}s")
    print(f"  Champion Checkpoint: {best_weights_path}")
    print(f"  Best Validation MPJPE: {best_val_mpjpe:.2f} mm")
    print(f"  Log Saved -> {log_csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
