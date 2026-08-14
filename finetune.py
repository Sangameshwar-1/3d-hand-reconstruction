"""
finetune.py — Stage 5b: Vision Transformer Fine-tuning.

Fine-tunes a Vision Transformer (ViT) on the paired dataset to correct Z-axis 
depth hallucination. We freeze the heavy backbone (feature extraction) and only 
train a new fully-connected regression head.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights

import config
from hand_dataset import HandPoseDataset


class FineTunedHandPoseModel(nn.Module):
    """
    Vision Transformer with a custom regression head for 3D hand pose.
    Backbone is frozen to allow training on standard hardware.
    """
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        
        # Load a pretrained ViT
        print("[MODEL] Loading pretrained ViT-B/16...")
        weights = ViT_B_16_Weights.DEFAULT
        self.backbone = vit_b_16(weights=weights)
        
        # Optionally freeze backbone
        if freeze_backbone:
            print("[MODEL] Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Replace the final classification head with our regression head
        # ViT-B hidden size is 768
        in_features = self.backbone.heads.head.in_features
        out_features = config.NUM_JOINTS * 3  # 63
        
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_features)
        )
        
        print(f"[MODEL] Initialized new regression head: {in_features} -> {out_features}")

    def forward(self, x):
        # Forward pass through modified ViT
        # Scale output by 200.0 to help gradients flow when regressing large absolute mm values
        return self.backbone(x) * 200.0


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss combining three terms:
      1. Anisotropic XYZ MSE      — penalises Z-axis errors more heavily.
      2. Bone-Length Regularizer   — keeps predicted bone lengths anatomically correct.
      3. Joint Angle Regularizer   — forbids accordion-fold (hyperextension) of finger hinges.
    """

    # Each triplet is (Base, Hinge, Tip) — one entry per inter-phalangeal hinge.
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

    def __init__(self, lambda_xy=1.0, lambda_z=10.0, lambda_bone=1.5, lambda_angle=2.0):
        super().__init__()
        self.lambda_xy = lambda_xy
        self.lambda_z = lambda_z
        self.lambda_bone = lambda_bone
        self.lambda_angle = lambda_angle
        self.mse = nn.MSELoss(reduction='none')

        # Extract bone connections from config
        self.bone_starts = [pair[0] for pair in config.SKELETON_CONNECTIONS]
        self.bone_ends   = [pair[1] for pair in config.SKELETON_CONNECTIONS]

    # ------------------------------------------------------------------
    # Loss term 3 — Joint Angle Regularizer (L_angle)
    # ------------------------------------------------------------------
    def calculate_angular_loss(self, predicted_joints: torch.Tensor) -> torch.Tensor:
        """
        Penalises non-physical accordion folding at each finger hinge.

        For every triplet (a, b, c) where b is the hinge:
          v1 = J_a - J_b   (vector pointing back to base)
          v2 = J_c - J_b   (vector pointing forward to tip)

        When the joint is straight  : cos(theta) = -1  (no penalty)
        When the joint is at 90 deg : cos(theta) =  0  (boundary)
        When the joint hyper-folds  : cos(theta) >  0  (PENALISED)

        Only violations above fold_threshold = 0.0 contribute to the loss
        via ReLU, so normal bending is never penalised.

        Args:
            predicted_joints: (B, 21, 3) tensor of predicted joint positions.

        Returns:
            Scalar angular loss averaged across all hinges and the batch.
        """
        fold_threshold = 0.0
        loss_angle = predicted_joints.new_zeros(1)  # device-agnostic zero

        for (a, b, c) in self.HINGE_TRIPLETS:
            v1 = predicted_joints[:, a, :] - predicted_joints[:, b, :]  # (B, 3)
            v2 = predicted_joints[:, c, :] - predicted_joints[:, b, :]  # (B, 3)

            # Normalise — add eps to avoid division by zero on degenerate bones
            v1_norm = F.normalize(v1, p=2, dim=1)
            v2_norm = F.normalize(v2, p=2, dim=1)

            # Dot product gives cos(theta) per sample  →  shape (B,)
            cos_theta = torch.sum(v1_norm * v2_norm, dim=1)

            # Only penalise values that exceed the biological threshold
            violation = torch.relu(cos_theta - fold_threshold)
            loss_angle = loss_angle + violation.mean()

        return loss_angle

    # ------------------------------------------------------------------
    # Combined forward pass
    # ------------------------------------------------------------------
    def forward(self, pred, target):
        pred_3d   = pred.view(-1, config.NUM_JOINTS, 3)
        target_3d = target.view(-1, config.NUM_JOINTS, 3)

        # 1. Anisotropic XYZ Loss
        se = self.mse(pred_3d, target_3d)          # (B, 21, 3)
        weights = torch.tensor(
            [self.lambda_xy, self.lambda_xy, self.lambda_z],
            dtype=se.dtype, device=se.device
        )
        xyz_loss = (se * weights.view(1, 1, 3)).mean()

        # 2. Bone-Length Regularization
        pred_starts   = pred_3d[:, self.bone_starts, :]
        pred_ends     = pred_3d[:, self.bone_ends,   :]
        pred_lengths  = torch.norm(pred_starts - pred_ends, dim=2)

        tgt_starts    = target_3d[:, self.bone_starts, :]
        tgt_ends      = target_3d[:, self.bone_ends,   :]
        tgt_lengths   = torch.norm(tgt_starts - tgt_ends, dim=2)

        bone_loss = torch.abs(pred_lengths - tgt_lengths).mean()

        # 3. Joint Angle Regularizer (accordion-fold prevention)
        angle_loss = self.calculate_angular_loss(pred_3d)

        total_loss = (
            xyz_loss
            + (self.lambda_bone  * bone_loss)
            + (self.lambda_angle * angle_loss)
        )
        return total_loss, xyz_loss.detach(), bone_loss.detach(), angle_loss.detach()


def train_model(dataset_dir: str, epochs: int, batch_size: int, lr: float):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRAIN] Using device: {device}")
    
    # 1. Datasets & Loaders
    train_dataset = HandPoseDataset(dataset_dir, split="train")
    val_dataset = HandPoseDataset(dataset_dir, split="val")
    
    if len(train_dataset) == 0:
        print("[ERROR] Training dataset is empty.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 2. Model, Loss, Optimizer
    model = FineTunedHandPoseModel(freeze_backbone=True).to(device)

    # Physics-Informed Loss: XYZ anisotropy + bone length + joint angle regularizer
    criterion = PhysicsInformedLoss(
        lambda_xy=1.0,
        lambda_z=10.0,
        lambda_bone=1.5,
        lambda_angle=2.0,
    )
    
    # Optimize ONLY the parameters that require gradients (our new head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    print("\n[TRAIN] Starting Fine-tuning...")
    print(f"  Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print("-" * 60)

    # --- CSV loss log (survives Ctrl-C; plot script reads this) ---
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.RESULTS_DIR / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss",
                         "xyz_loss", "bone_loss", "angle_loss", "mpjpe_mm"])
    print(f"[TRAIN] Loss log -> {log_path}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for imgs, joints in train_loader:
            imgs = imgs.to(device)
            joints = joints.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss, _, _, _ = criterion(outputs, joints)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            
        train_loss = train_loss / len(train_dataset)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        
        # For MPJPE calculation during validation
        # outputs and joints are flattened (B, 63). We reshape to (B, 21, 3)
        # to compute standard euclidean distance.
        total_mpjpe = 0.0
        
        with torch.no_grad():
            for imgs, joints in val_loader:
                imgs = imgs.to(device)
                joints = joints.to(device)
                
                outputs = model(imgs)
                loss, v_xyz, v_bone, v_angle = criterion(outputs, joints)
                val_loss += loss.item() * imgs.size(0)

                # Compute MPJPE
                out_reshaped = outputs.view(-1, 21, 3)
                gt_reshaped = joints.view(-1, 21, 3)
                
                # euclidean distance per joint, then mean
                dists = torch.norm(out_reshaped - gt_reshaped, dim=2) # (B, 21)
                batch_mpjpe = torch.mean(dists).item()
                total_mpjpe += batch_mpjpe * imgs.size(0)
                
        val_loss  = val_loss  / len(val_dataset)
        val_mpjpe = total_mpjpe / len(val_dataset)

        print(
            f"Epoch {epoch+1:03d}/{epochs:03d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"XYZ: {v_xyz.item():.4f}  Bone: {v_bone.item():.4f}  Angle: {v_angle.item():.4f} | "
            f"MPJPE: {val_mpjpe:.2f} mm"
        )

        # Append row to CSV log
        csv_writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{v_xyz.item():.6f}",
            f"{v_bone.item():.6f}",
            f"{v_angle.item():.6f}",
            f"{val_mpjpe:.4f}",
        ])
        log_file.flush()  # safe against interruptions
              
        # --- Save Best Model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = config.CHECKPOINTS_DIR / "best_model.pth"
            config.CHECKPOINTS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved: {save_path.name}  (val={val_loss:.4f})")
            
    log_file.close()
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"[TRAIN] Complete in {total_time/60:.1f} minutes.")
    print(f"  Best Validation MSE : {best_val_loss:.4f}")
    print(f"  Loss log saved to   : {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 5b: Fine-tune ViT model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    args = parser.parse_args()

    train_model(args.dataset, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
