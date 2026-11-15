"""Independent test-set evaluation for ResNet50 Classifier."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.classifier import BrainTumorClassifier
from src.data.classification_dataset import BrainTumorClassificationDataset
from src.utils.metrics import calculate_classification_metrics
from src.utils.seed import get_device
from src.utils.visualization import plot_confusion_matrix


def evaluate_classifier_model(
    checkpoint_path: str = "checkpoints/best_classifier.pth",
    config_path: str = "config/config.yaml",
    test_dir: Optional[str] = None,
    output_dir: str = "results/classification",
    device_pref: str = "auto",
) -> Dict:
    """Evaluate a saved classifier checkpoint on the test set."""
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required for evaluation.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(device_pref)
    test_path = test_dir or config.get("dataset", {}).get("classification", {}).get("test_dir", "data/brisc2025/classification_task/test")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not Path(test_path).exists():
        print(f"[!] Warning: Test dataset directory not found at: {test_path}")
        return {"status": "test_dataset_missing", "path": test_path}

    if not Path(checkpoint_path).exists():
        print(f"[!] Warning: Model checkpoint not found at: {checkpoint_path}")
        return {"status": "checkpoint_missing", "path": checkpoint_path}

    # Load Data
    dataset_manager = BrainTumorClassificationDataset(
        train_dir=test_path,  # dummy reference
        test_dir=test_path,
        image_size=tuple(config.get("dataset", {}).get("image_size", [224, 224])),
    )
    _, _, test_loader = dataset_manager.get_dataloaders(
        batch_size=config.get("classification", {}).get("batch_size", 32),
        num_workers=config.get("dataset", {}).get("num_workers", 2),
    )

    if test_loader is None or len(test_loader) == 0:
        print(f"[!] Error: No samples found in {test_path}")
        return {"status": "empty_test_set"}

    class_names = dataset_manager.class_names
    num_classes = len(class_names)

    # Load Model
    model = BrainTumorClassifier(num_classes=num_classes, pretrained=False, class_names=class_names)
    model.load_checkpoint(checkpoint_path, device=str(device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print(f"[*] Evaluating classifier on {len(test_loader.dataset)} test samples...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    # Calculate metrics
    metrics = calculate_classification_metrics(all_targets, all_preds, class_names=class_names)

    print("\n" + "=" * 60)
    print("CLASSIFICATION TEST SET EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:    {metrics['macro_f1']:.4f}")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 60)
    for c_name, c_metrics in metrics["per_class"].items():
        print(
            f"{c_name:<15} "
            f"{c_metrics['precision']:<12.4f} "
            f"{c_metrics['recall']:<12.4f} "
            f"{c_metrics['f1_score']:<12.4f} "
            f"{c_metrics['support']:<8d}"
        )
    print("=" * 60)

    # Save metrics JSON
    metrics_file = out_path / "test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Saved evaluation metrics to {metrics_file}")

    # Plot & Save Confusion Matrix
    cm_plot_path = out_path / "confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names,
        save_path=str(cm_plot_path),
        title="Test Set Confusion Matrix (ResNet50)",
    )
    print(f"[+] Saved confusion matrix plot to {cm_plot_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 Classifier")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_classifier.pth", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--test-dir", type=str, default=None, help="Path to test images")
    parser.add_argument("--output-dir", type=str, default="results/classification", help="Output directory")
    args = parser.parse_args()

    evaluate_classifier_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
    )
