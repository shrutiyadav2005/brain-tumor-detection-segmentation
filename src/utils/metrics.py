"""Evaluation metrics for classification and segmentation tasks.

Implements pure-Python mathematical fallbacks when NumPy is not installed in the
environment, ensuring tests and utilities can execute in minimal environments.
"""

from typing import Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ==============================================================================
# CLASSIFICATION METRICS
# ==============================================================================

def calculate_classification_metrics(
    y_true: Union[List[int], "np.ndarray"],
    y_pred: Union[List[int], "np.ndarray"],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict[str, float], List[List[int]]]]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        class_names: List of human-readable class names.

    Returns:
        Dictionary containing accuracy, macro precision, recall, f1,
        per-class reports, and confusion matrix.
    """
    if NUMPY_AVAILABLE and hasattr(y_true, "tolist"):
        y_true_list = y_true.tolist()
    else:
        y_true_list = list(y_true)

    if NUMPY_AVAILABLE and hasattr(y_pred, "tolist"):
        y_pred_list = y_pred.tolist()
    else:
        y_pred_list = list(y_pred)

    if len(y_true_list) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }

    unique_classes = sorted(list(set(y_true_list).union(set(y_pred_list))))
    num_classes = len(unique_classes) if not class_names else len(class_names)

    # Initialize confusion matrix [num_classes x num_classes]
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true_list, y_pred_list):
        t, p = int(t), int(p)
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    total_samples = len(y_true_list)
    correct_samples = sum(cm[i][i] for i in range(num_classes))
    accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = float(cm[c][c])
        fp = float(sum(cm[i][c] for i in range(num_classes)) - tp)
        fn = float(sum(cm[c][j] for j in range(num_classes)) - tp)
        support = int(sum(cm[c][j] for j in range(num_classes)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        c_name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        per_class[c_name] = {
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1, 4)),
            "support": support,
        }

    macro_precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    macro_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0

    return {
        "accuracy": float(round(accuracy, 4)),
        "macro_precision": float(round(macro_precision, 4)),
        "macro_recall": float(round(macro_recall, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


# ==============================================================================
# SEGMENTATION METRICS
# ==============================================================================

def _flatten_mask(mask) -> List[float]:
    """Helper to flatten 2D/3D array or list into a 1D float list."""
    if NUMPY_AVAILABLE and isinstance(mask, np.ndarray):
        return mask.flatten().astype(float).tolist()
    if isinstance(mask, list):
        flat = []
        for item in mask:
            if isinstance(item, list):
                flat.extend(_flatten_mask(item))
            else:
                flat.append(float(item))
        return flat
    return [float(mask)]


def calculate_dice_coefficient(
    y_pred,
    y_true,
    smooth: float = 1e-6,
    threshold: float = 0.5,
) -> float:
    """Compute the Dice Similarity Coefficient (DSC) between binary masks.

    DSC = (2 * |A ∩ B|) / (|A| + |B|)
    """
    p_flat = _flatten_mask(y_pred)
    t_flat = _flatten_mask(y_true)

    p_bin = [1.0 if v >= threshold else 0.0 for v in p_flat]
    t_bin = [1.0 if v >= 0.5 else 0.0 for v in t_flat]

    intersection = sum(p * t for p, t in zip(p_bin, t_bin))
    cardinality = sum(p_bin) + sum(t_bin)

    if cardinality == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return float(max(0.0, min(1.0, dice)))


def calculate_iou(
    y_pred,
    y_true,
    smooth: float = 1e-6,
    threshold: float = 0.5,
) -> float:
    """Compute the Intersection over Union (IoU / Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|
    """
    p_flat = _flatten_mask(y_pred)
    t_flat = _flatten_mask(y_true)

    p_bin = [1.0 if v >= threshold else 0.0 for v in p_flat]
    t_bin = [1.0 if v >= 0.5 else 0.0 for v in t_flat]

    intersection = sum(p * t for p, t in zip(p_bin, t_bin))
    union = sum(p_bin) + sum(t_bin) - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = (intersection + smooth) / (union + smooth)
    return float(max(0.0, min(1.0, iou)))


def calculate_segmentation_metrics(
    y_pred,
    y_true,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Calculate all standard medical segmentation metrics."""
    p_flat = _flatten_mask(y_pred)
    t_flat = _flatten_mask(y_true)

    p_bin = [1.0 if v >= threshold else 0.0 for v in p_flat]
    t_bin = [1.0 if v >= 0.5 else 0.0 for v in t_flat]

    tp = sum(p * t for p, t in zip(p_bin, t_bin))
    fp = sum(p * (1.0 - t) for p, t in zip(p_bin, t_bin))
    fn = sum((1.0 - p) * t for p, t in zip(p_bin, t_bin))
    tn = sum((1.0 - p) * (1.0 - t) for p, t in zip(p_bin, t_bin))
    total = float(len(p_bin))

    dice = calculate_dice_coefficient(y_pred, y_true, smooth=smooth, threshold=threshold)
    iou = calculate_iou(y_pred, y_true, smooth=smooth, threshold=threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    pixel_acc = (tp + tn) / total if total > 0 else 0.0

    return {
        "dice_coefficient": float(round(dice, 4)),
        "iou": float(round(iou, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "pixel_accuracy": float(round(pixel_acc, 4)),
    }
