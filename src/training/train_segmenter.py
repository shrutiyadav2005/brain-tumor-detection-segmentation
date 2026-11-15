"""Training script for U-Net Brain Tumor Segmentation."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.segmenter import BrainTumorSegmenter
from src.data.segmentation_dataset import get_segmentation_dataloaders
from src.utils.metrics import calculate_dice_coefficient, calculate_iou
from src.utils.seed import get_device, set_seed
from src.utils.visualization import plot_training_history


# ==============================================================================
# SEGMENTATION LOSS FUNCTIONS
# ==============================================================================

class DiceLoss(nn.Module if TORCH_AVAILABLE else object):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class CombinedBceDiceLoss(nn.Module if TORCH_AVAILABLE else object):
    """Hybrid loss combining BCEWithLogitsLoss and Soft Dice Loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


# ==============================================================================
# TRAINING & VALIDATION LOOPS
# ==============================================================================

def train_segmentation_epoch(
    model: "nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    criterion: "nn.Module",
    optimizer: "optim.Optimizer",
    device: "torch.device",
) -> Tuple[float, float, float]:
    """Execute one training epoch for segmentation."""
    model.train()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training Seg", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute batch metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = masks.cpu().numpy()
            batch_dice = calculate_dice_coefficient(probs, targets_np)
            batch_iou = calculate_iou(probs, targets_np)

        total_dice += batch_dice
        total_iou += batch_iou
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{batch_dice:.4f}"})

    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
    epoch_dice = total_dice / num_batches if num_batches > 0 else 0.0
    epoch_iou = total_iou / num_batches if num_batches > 0 else 0.0
    return epoch_loss, epoch_dice, epoch_iou


def validate_segmentation_epoch(
    model: "nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    criterion: "nn.Module",
    device: "torch.device",
) -> Tuple[float, float, float]:
    """Execute one validation epoch for segmentation."""
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = masks.cpu().numpy()
            total_dice += calculate_dice_coefficient(probs, targets_np)
            total_iou += calculate_iou(probs, targets_np)
            num_batches += 1

    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
    epoch_dice = total_dice / num_batches if num_batches > 0 else 0.0
    epoch_iou = total_iou / num_batches if num_batches > 0 else 0.0
    return epoch_loss, epoch_dice, epoch_iou


def run_segmenter_training(config_path: str = "config/config.yaml", **kwargs) -> Dict:
    """Run full U-Net segmentation training pipeline."""
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required for segmentation training.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg_gen = config.get("general", {})
    cfg_data = config.get("dataset", {})
    cfg_seg = config.get("segmentation", {})

    seed = kwargs.get("seed", cfg_gen.get("seed", 42))
    set_seed(seed)

    device = get_device(kwargs.get("device", cfg_gen.get("device", "auto")))
    print(f"[*] Training U-Net Segmenter on device: {device}")

    # Paths
    train_images = kwargs.get("train_images", cfg_data.get("segmentation", {}).get("train_images", "data/brisc2025/segmentation_task/train/images"))
    train_masks = kwargs.get("train_masks", cfg_data.get("segmentation", {}).get("train_masks", "data/brisc2025/segmentation_task/train/masks"))
    checkpoint_dir = Path(kwargs.get("checkpoint_dir", cfg_seg.get("checkpoint_dir", "checkpoints")))
    results_dir = Path(kwargs.get("results_dir", cfg_seg.get("results_dir", "results/segmentation")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not (Path(train_images).exists() and Path(train_masks).exists()):
        print(f"[!] Warning: Segmentation dataset paths not found:\n    Images: {train_images}\n    Masks: {train_masks}")
        print("    Please place BRISC2025 segmentation data under data/brisc2025/segmentation_task/train/")
        return {"status": "dataset_missing"}

    # DataLoaders
    batch_size = kwargs.get("batch_size", cfg_seg.get("batch_size", 16))
    image_size = tuple(cfg_data.get("image_size", [224, 224]))
    val_split = cfg_data.get("segmentation", {}).get("val_split", 0.15)
    test_split = cfg_data.get("segmentation", {}).get("test_split", 0.15)

    train_loader, val_loader, _ = get_segmentation_dataloaders(
        train_images=train_images,
        train_masks=train_masks,
        image_size=image_size,
        val_split=val_split,
        test_split=test_split,
        batch_size=batch_size,
        seed=seed,
    )

    if train_loader is None or len(train_loader) == 0:
        print("[!] Error: No segmentation samples loaded.")
        return {"status": "empty_dataset"}

    print(f"[*] Segmentation dataset: {len(train_loader.dataset)} train pairs, {len(val_loader.dataset) if val_loader else 0} val pairs")

    # Model & Loss
    encoder_name = cfg_seg.get("encoder_name", "resnet34")
    encoder_weights = cfg_seg.get("encoder_weights", "imagenet")
    model = BrainTumorSegmenter(encoder_name=encoder_name, encoder_weights=encoder_weights)
    model = model.to(device)

    loss_type = cfg_seg.get("loss_type", "combined")
    if loss_type == "combined":
        criterion = CombinedBceDiceLoss()
    elif loss_type == "dice":
        criterion = DiceLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    lr = kwargs.get("learning_rate", cfg_seg.get("learning_rate", 0.0005))
    weight_decay = cfg_seg.get("weight_decay", 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = kwargs.get("epochs", cfg_seg.get("epochs", 20))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg_seg.get("lr_step_size", 7), gamma=cfg_seg.get("lr_gamma", 0.5))
    patience = cfg_seg.get("early_stopping_patience", 6)

    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_val_dice = -1.0
    patience_counter = 0
    best_checkpoint_path = checkpoint_dir / cfg_seg.get("checkpoint_name", "best_segmenter.pth")

    print(f"[*] Starting U-Net training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_dice, train_iou = train_segmentation_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_segmentation_epoch(model, val_loader, criterion, device) if val_loader else (0.0, 0.0, 0.0)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_dice"].append(round(train_dice, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_dice"].append(round(val_dice, 4))
        history["val_iou"].append(round(val_iou, 4))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
        )

        # Save best model based on validation Dice score
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                best_checkpoint_path,
            )
            print(f"  [+] Saved new best segmenter checkpoint to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered after {epoch} epochs.")
                break

    # Save History & Curves
    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    curves_path = results_dir / "training_curves.png"
    plot_training_history(history, save_path=str(curves_path), title="U-Net Segmentation Training History")
    print(f"[*] Segmentation history saved to {history_path}")
    print(f"[*] Curves plotted to {curves_path}")

    return {"status": "success", "history": history, "best_checkpoint": str(best_checkpoint_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Tumor Segmenter")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    run_segmenter_training(config_path=args.config, **kwargs)
