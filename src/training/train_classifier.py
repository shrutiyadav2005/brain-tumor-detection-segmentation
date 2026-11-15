"""Training script for ResNet50 4-Class Brain MRI Classifier."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.classifier import BrainTumorClassifier
from src.data.classification_dataset import BrainTumorClassificationDataset
from src.utils.seed import get_device, set_seed
from src.utils.visualization import plot_training_history


def train_epoch(
    model: "nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    criterion: "nn.Module",
    optimizer: "optim.Optimizer",
    device: "torch.device",
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_epoch(
    model: "nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    criterion: "nn.Module",
    device: "torch.device",
) -> Tuple[float, float]:
    """Execute one validation epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def run_classifier_training(config_path: str = "config/config.yaml", **kwargs) -> Dict:
    """Run full classifier training routine."""
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch and torchvision are required for training.")
        sys.exit(1)

    # 1. Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg_gen = config.get("general", {})
    cfg_data = config.get("dataset", {})
    cfg_cls = config.get("classification", {})

    seed = kwargs.get("seed", cfg_gen.get("seed", 42))
    set_seed(seed)

    device_str = kwargs.get("device", cfg_gen.get("device", "auto"))
    device = get_device(device_str)
    print(f"[*] Training ResNet50 Classifier on device: {device}")

    # 2. Paths
    train_dir = kwargs.get("train_dir", cfg_data.get("classification", {}).get("train_dir", "data/brisc2025/classification_task/train"))
    test_dir = kwargs.get("test_dir", cfg_data.get("classification", {}).get("test_dir", "data/brisc2025/classification_task/test"))
    checkpoint_dir = Path(kwargs.get("checkpoint_dir", cfg_cls.get("checkpoint_dir", "checkpoints")))
    results_dir = Path(kwargs.get("results_dir", cfg_cls.get("results_dir", "results/classification")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not Path(train_dir).exists():
        print(f"[!] Warning: Training dataset directory not found at: {train_dir}")
        print("    Please place the BRISC2025 classification dataset in data/brisc2025/classification_task/train")
        return {"status": "dataset_missing", "path": train_dir}

    # 3. DataLoaders
    batch_size = kwargs.get("batch_size", cfg_cls.get("batch_size", 32))
    image_size = tuple(cfg_data.get("image_size", [224, 224]))
    val_split = cfg_data.get("classification", {}).get("val_split_from_train", 0.15)
    num_workers = cfg_data.get("num_workers", 2)

    dataset_manager = BrainTumorClassificationDataset(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=image_size,
        val_split=val_split,
        seed=seed,
    )
    train_loader, val_loader, _ = dataset_manager.get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if train_loader is None or len(train_loader) == 0:
        print(f"[!] Error: No training samples loaded from {train_dir}")
        return {"status": "empty_dataset"}

    print(f"[*] Dataset loaded: {len(train_loader.dataset)} train samples, {len(val_loader.dataset) if val_loader else 0} val samples")
    print(f"[*] Categories: {dataset_manager.class_names}")

    # 4. Model, Criterion, Optimizer, Scheduler
    num_classes = cfg_cls.get("num_classes", len(dataset_manager.class_names))
    pretrained = cfg_cls.get("pretrained", True)
    model = BrainTumorClassifier(num_classes=num_classes, pretrained=pretrained, class_names=dataset_manager.class_names)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    lr = kwargs.get("learning_rate", cfg_cls.get("learning_rate", 0.0005))
    weight_decay = cfg_cls.get("weight_decay", 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = kwargs.get("epochs", cfg_cls.get("epochs", 15))
    step_size = cfg_cls.get("lr_step_size", 5)
    gamma = cfg_cls.get("lr_gamma", 0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    patience = cfg_cls.get("early_stopping_patience", 5)

    # 5. Training Loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = checkpoint_dir / cfg_cls.get("checkpoint_name", "best_classifier.pth")

    print(f"[*] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device) if val_loader else (0.0, 0.0)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": dataset_manager.class_names,
                },
                best_checkpoint_path,
            )
            print(f"  [+] Saved new best model checkpoint to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered after {epoch} epochs.")
                break

    # 6. Save Training History & Curves
    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    curves_path = results_dir / "training_curves.png"
    plot_training_history(history, save_path=str(curves_path), title="ResNet50 Classification Training History")
    print(f"[*] Training history saved to {history_path}")
    print(f"[*] Curves plotted to {curves_path}")

    return {"status": "success", "history": history, "best_checkpoint": str(best_checkpoint_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 Brain Tumor Classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    kwargs = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    run_classifier_training(config_path=args.config, **kwargs)
