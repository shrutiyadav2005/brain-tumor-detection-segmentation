"""Independent test-set evaluation for U-Net Tumor Segmenter."""

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

from models.segmenter import BrainTumorSegmenter
from src.data.segmentation_dataset import get_segmentation_dataloaders
from src.utils.metrics import calculate_segmentation_metrics
from src.utils.seed import get_device
from src.utils.visualization import plot_segmentation_comparison


def evaluate_segmenter_model(
    checkpoint_path: str = "checkpoints/best_segmenter.pth",
    config_path: str = "config/config.yaml",
    test_images: Optional[str] = None,
    test_masks: Optional[str] = None,
    output_dir: str = "results/segmentation",
    device_pref: str = "auto",
    num_visualizations: int = 5,
) -> Dict:
    """Evaluate trained U-Net segmentation checkpoint on test set."""
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required for segmentation evaluation.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(device_pref)
    cfg_seg = config.get("segmentation", {})
    cfg_data = config.get("dataset", {})

    test_img_dir = test_images or cfg_data.get("segmentation", {}).get("test_images", "data/brisc2025/segmentation_task/test/images")
    test_msk_dir = test_masks or cfg_data.get("segmentation", {}).get("test_masks", "data/brisc2025/segmentation_task/test/masks")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    vis_dir = out_path / "sample_predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not (Path(test_img_dir).exists() and Path(test_msk_dir).exists()):
        print(f"[!] Warning: Test dataset paths not found:\n    Images: {test_img_dir}\n    Masks: {test_msk_dir}")
        return {"status": "test_dataset_missing"}

    if not Path(checkpoint_path).exists():
        print(f"[!] Warning: Model checkpoint not found at: {checkpoint_path}")
        return {"status": "checkpoint_missing", "path": checkpoint_path}

    _, _, test_loader = get_segmentation_dataloaders(
        train_images=test_img_dir,
        train_masks=test_msk_dir,
        test_images=test_img_dir,
        test_masks=test_msk_dir,
        image_size=tuple(cfg_data.get("image_size", [224, 224])),
        batch_size=cfg_seg.get("batch_size", 16),
    )

    if test_loader is None or len(test_loader) == 0:
        print(f"[!] Error: No segmentation samples loaded from {test_img_dir}")
        return {"status": "empty_test_set"}

    # Load Model
    model = BrainTumorSegmenter(
        encoder_name=cfg_seg.get("encoder_name", "resnet34"),
        encoder_weights=None,
    )
    model.load_checkpoint(checkpoint_path, device=str(device))
    model = model.to(device)
    model.eval()

    all_dices = []
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []

    saved_vis_count = 0
    threshold = cfg_seg.get("threshold", 0.5)

    print(f"[*] Evaluating U-Net Segmenter on {len(test_loader.dataset)} test samples...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = masks.cpu().numpy()

            for i in range(images.size(0)):
                p_mask = probs[i, 0]
                t_mask = targets_np[i, 0]

                sample_metrics = calculate_segmentation_metrics(p_mask, t_mask, threshold=threshold)
                all_dices.append(sample_metrics["dice_coefficient"])
                all_ious.append(sample_metrics["iou"])
                all_precisions.append(sample_metrics["precision"])
                all_recalls.append(sample_metrics["recall"])
                all_accuracies.append(sample_metrics["pixel_accuracy"])

                # Save sample visual comparisons
                if saved_vis_count < num_visualizations:
                    # Un-normalize image for plotting
                    raw_img = images[i].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    vis_img = np.clip(raw_img * std + mean, 0.0, 1.0)

                    vis_save_path = vis_dir / f"test_sample_{saved_vis_count + 1}.png"
                    plot_segmentation_comparison(
                        image=vis_img,
                        true_mask=t_mask,
                        pred_mask=(p_mask >= threshold).astype(np.float32),
                        save_path=str(vis_save_path),
                        dice_score=sample_metrics["dice_coefficient"],
                        iou_score=sample_metrics["iou"],
                    )
                    saved_vis_count += 1

    summary_metrics = {
        "mean_dice_coefficient": float(round(np.mean(all_dices), 4)),
        "mean_iou": float(round(np.mean(all_ious), 4)),
        "mean_precision": float(round(np.mean(all_precisions), 4)),
        "mean_recall": float(round(np.mean(all_recalls), 4)),
        "mean_pixel_accuracy": float(round(np.mean(all_accuracies), 4)),
        "num_test_samples": len(all_dices),
        "threshold": threshold,
    }

    print("\n" + "=" * 60)
    print("SEGMENTATION TEST SET EVALUATION REPORT")
    print("=" * 60)
    print(f"Mean Dice Coefficient:  {summary_metrics['mean_dice_coefficient']:.4f}")
    print(f"Mean IoU (Jaccard):     {summary_metrics['mean_iou']:.4f}")
    print(f"Mean Precision:         {summary_metrics['mean_precision']:.4f}")
    print(f"Mean Recall:            {summary_metrics['mean_recall']:.4f}")
    print(f"Mean Pixel Accuracy:    {summary_metrics['mean_pixel_accuracy'] * 100:.2f}%")
    print(f"Evaluated Samples:      {summary_metrics['num_test_samples']}")
    print("=" * 60)

    # Save metrics JSON
    metrics_file = out_path / "test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"[+] Saved evaluation metrics to {metrics_file}")
    print(f"[+] Saved {saved_vis_count} sample visual comparisons to {vis_dir}")

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net Tumor Segmenter")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_segmenter.pth")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--test-images", type=str, default=None)
    parser.add_argument("--test-masks", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/segmentation")
    args = parser.parse_args()

    evaluate_segmenter_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        test_images=args.test_images,
        test_masks=args.test_masks,
        output_dir=args.output_dir,
    )
