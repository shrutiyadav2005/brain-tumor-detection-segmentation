"""Visualization utilities for classification, segmentation, and explainability."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless execution / server environments
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training and Validation History",
) -> None:
    """Plot and save training and validation metrics curves over epochs.

    Args:
        history: Dictionary containing lists of epoch metrics (e.g. train_loss, val_loss).
        save_path: File path where the plot figure should be saved.
        title: Overall super-title for the figure.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if not epochs:
        return

    has_acc = "train_acc" in history
    has_dice = "train_dice" in history or "val_dice" in history

    num_subplots = 2 if (has_acc or has_dice) else 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 5))
    if num_subplots == 1:
        axes = [axes]

    # Subplot 1: Loss curves
    axes[0].plot(epochs, history.get("train_loss", []), "b-o", label="Train Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    axes[0].set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Subplot 2: Accuracy or Dice
    if has_acc:
        axes[1].plot(epochs, history.get("train_acc", []), "g-o", label="Train Acc", linewidth=2)
        if "val_acc" in history and history["val_acc"]:
            axes[1].plot(epochs, history["val_acc"], "m--s", label="Val Acc", linewidth=2)
        axes[1].set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(loc="lower right")
    elif has_dice:
        if "train_dice" in history:
            axes[1].plot(epochs, history["train_dice"], "g-o", label="Train Dice", linewidth=2)
        if "val_dice" in history and history["val_dice"]:
            axes[1].plot(epochs, history["val_dice"], "m--s", label="Val Dice", linewidth=2)
        axes[1].set_title("Dice Similarity over Epochs", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Dice Score", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(loc="lower right")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a stylized confusion matrix.

    Args:
        cm: 2D confusion matrix array.
        class_names: List of class labels corresponding to matrix rows/cols.
        save_path: File path to save figure.
        title: Title of plot.
    """
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))

    cax = ax.matshow(cm_arr, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="left", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate numbers in cells
    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary segmentation mask onto an RGB MRI image with transparency.

    Args:
        image: Original RGB image normalized to [0, 1] or uint8 in [0, 255].
        mask: Binary mask with values in {0, 1} or probabilities in [0, 1].
        color: RGB tuple for highlighting tumor region (default: bright red).
        alpha: Blending transparency factor.

    Returns:
        Blended RGB image array normalized in [0, 1].
    """
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    msk = (np.asarray(mask) >= 0.5).astype(np.float32)
    if msk.ndim == 3 and msk.shape[-1] == 1:
        msk = msk.squeeze(-1)

    overlay = img.copy()
    for c in range(3):
        overlay[..., c] = np.where(msk > 0, (1 - alpha) * img[..., c] + alpha * color[c], img[..., c])

    return np.clip(overlay, 0.0, 1.0)


def plot_segmentation_comparison(
    image: np.ndarray,
    true_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    save_path: Optional[str] = None,
    dice_score: Optional[float] = None,
    iou_score: Optional[float] = None,
) -> None:
    """Plot side-by-side comparison for tumor segmentation evaluation.

    Args:
        image: Original MRI image.
        true_mask: Ground-truth mask (optional).
        pred_mask: Predicted segmentation mask.
        save_path: Destination path for saving plot.
        dice_score: Optional Dice score to display in title.
        iou_score: Optional IoU score to display in title.
    """
    has_gt = true_mask is not None
    cols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, cols, figsize=(4.5 * cols, 4.5))

    img_norm = np.asarray(image, dtype=np.float32)
    if img_norm.max() > 1.0:
        img_norm = img_norm / 255.0

    # 1. Original MRI
    axes[0].imshow(img_norm, cmap="gray" if img_norm.ndim == 2 else None)
    axes[0].set_title("Original MRI Scan", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    idx = 1
    # 2. Ground Truth Mask (if available)
    if has_gt:
        axes[idx].imshow(true_mask, cmap="gray")
        axes[idx].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[idx].axis("off")
        idx += 1

    # 3. Predicted Mask
    axes[idx].imshow(pred_mask, cmap="inferno")
    axes[idx].set_title("Predicted Mask", fontsize=11, fontweight="bold")
    axes[idx].axis("off")
    idx += 1

    # 4. Color Overlay
    overlay = create_segmentation_overlay(img_norm, pred_mask)
    axes[idx].imshow(overlay)
    title_suffix = ""
    if dice_score is not None and iou_score is not None:
        title_suffix = f"\n(Dice: {dice_score:.3f} | IoU: {iou_score:.3f})"
    axes[idx].set_title(f"Tumor Overlay{title_suffix}", fontsize=11, fontweight="bold")
    axes[idx].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
