"""Data preprocessing, augmentation pipelines, and dataset integrity verification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

# Standard ImageNet statistics used by torchvision pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ==============================================================================
# TRANSFORMATION PIPELINES
# ==============================================================================

def get_classification_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
):
    """Get PyTorch transformation pipeline for classification.

    Applies ImageNet normalization required for ResNet50. Training mode adds
    mild medical imaging data augmentations (horizontal flip, slight rotation).

    Args:
        image_size: Target height and width (H, W).
        is_training: If True, applies data augmentations.

    Returns:
        torchvision.transforms.Compose instance.
    """
    from torchvision import transforms

    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def preprocess_image_for_inference(
    image: Union["Image.Image", "np.ndarray", str],
    image_size: Tuple[int, int] = (224, 224),
):
    """Standardize and preprocess a single input image for model inference.

    Args:
        image: PIL Image, NumPy array, or path to image.
        image_size: Model input dimensions (H, W).

    Returns:
        Tuple of (preprocessed_tensor of shape [1, 3, H, W], original_pil_image).
    """
    import torch
    from torchvision import transforms

    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3:
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Unsupported numpy image array shape: {image.shape}")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    transform = get_classification_transforms(image_size=image_size, is_training=False)
    tensor = transform(pil_img).unsqueeze(0)  # Shape: [1, 3, H, W]
    return tensor, pil_img


def preprocess_mask(
    mask_input: Union["Image.Image", "np.ndarray", str],
    target_size: Tuple[int, int] = (224, 224),
) -> Union["np.ndarray", List[List[float]]]:
    """Preprocess a ground-truth or target segmentation mask.

    CRITICAL: Enforces NEAREST-NEIGHBOR interpolation to prevent partial-volume
    filtering from introducing continuous floating-point values at binary borders.

    Args:
        mask_input: PIL image, file path, or numpy array.
        target_size: Target height and width (H, W).

    Returns:
        2D NumPy binary float array with values strictly in {0.0, 1.0} of shape target_size.
    """
    if isinstance(mask_input, str):
        mask_img = Image.open(mask_input).convert("L")
    elif isinstance(mask_input, np.ndarray):
        if mask_input.ndim == 3:
            mask_input = mask_input[..., 0]
        mask_img = Image.fromarray(mask_input.astype(np.uint8)).convert("L")
    elif isinstance(mask_input, Image.Image):
        mask_img = mask_input.convert("L")
    else:
        raise TypeError(f"Unsupported mask type: {type(mask_input)}")

    # Enforce Nearest-Neighbor interpolation for mask resizing
    mask_resized = mask_img.resize(target_size, resample=Image.Resampling.NEAREST)
    mask_arr = np.array(mask_resized, dtype=np.float32)

    # Strictly binarize: any nonzero value is considered tumor
    binary_mask = (mask_arr > 0).astype(np.float32)
    return binary_mask


# ==============================================================================
# DATASET INTEGRITY & VALIDATION
# ==============================================================================

def verify_dataset_integrity(
    root_dir: str = "data/brisc2025",
) -> Dict[str, Union[bool, Dict[str, int], List[str]]]:
    """Scan and verify dataset integrity for BRISC2025 tasks.

    Checks:
        1. Existence of classification train and test folders.
        2. Per-class sample counts.
        3. Existence of segmentation images and masks.
        4. Filename correspondence and pairing between images and masks.
        5. Orphaned images (no matching mask) or orphaned masks (no matching image).

    Args:
        root_dir: Root directory of the dataset.

    Returns:
        Summary dictionary with verification status and statistics.
    """
    root = Path(root_dir)
    results = {
        "dataset_found": False,
        "classification": {"train_counts": {}, "test_counts": {}, "is_valid": False},
        "segmentation": {"train_pairs": 0, "orphaned_images": [], "orphaned_masks": [], "is_valid": False},
        "errors": [],
        "warnings": [],
    }

    if not root.exists():
        results["errors"].append(f"Dataset root directory does not exist: {root.resolve()}")
        return results

    results["dataset_found"] = True

    # 1. Classification verification
    class_dir = root / "classification_task"
    if class_dir.exists():
        for split in ["train", "test"]:
            split_dir = class_dir / split
            counts = {}
            if split_dir.exists():
                for class_folder in sorted(split_dir.iterdir()):
                    if class_folder.is_dir() and not class_folder.name.startswith("."):
                        valid_files = [
                            f for f in class_folder.iterdir()
                            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not f.name.startswith(".")
                        ]
                        counts[class_folder.name] = len(valid_files)
            results["classification"][f"{split}_counts"] = counts

        train_c = results["classification"]["train_counts"]
        if train_c and all(v > 0 for v in train_c.values()):
            results["classification"]["is_valid"] = True
    else:
        results["warnings"].append("No classification_task folder found under root.")

    # 2. Segmentation verification
    seg_dir = root / "segmentation_task"
    if seg_dir.exists():
        train_img_dir = seg_dir / "train" / "images"
        train_mask_dir = seg_dir / "train" / "masks"

        if train_img_dir.exists() and train_mask_dir.exists():
            img_files = {
                f.stem: f for f in train_img_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not f.name.startswith(".")
            }
            mask_files = {
                f.stem: f for f in train_mask_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not f.name.startswith(".")
            }

            matched_stems = set(img_files.keys()).intersection(set(mask_files.keys()))
            orphaned_imgs = set(img_files.keys()) - set(mask_files.keys())
            orphaned_msks = set(mask_files.keys()) - set(img_files.keys())

            results["segmentation"]["train_pairs"] = len(matched_stems)
            results["segmentation"]["orphaned_images"] = sorted(list(orphaned_imgs))[:10]
            results["segmentation"]["orphaned_masks"] = sorted(list(orphaned_msks))[:10]

            if len(matched_stems) > 0 and len(orphaned_imgs) == 0:
                results["segmentation"]["is_valid"] = True
            elif len(matched_stems) > 0:
                results["warnings"].append(
                    f"Segmentation: Found {len(matched_stems)} matched pairs, but {len(orphaned_imgs)} "
                    f"images lack matching masks."
                )
                results["segmentation"]["is_valid"] = True
            else:
                results["errors"].append("Segmentation: Zero matched image-mask pairs found in train folder.")
        else:
            results["warnings"].append("Segmentation train/images or train/masks folders not found.")
    else:
        results["warnings"].append("No segmentation_task folder found under root.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify BRISC2025 dataset integrity")
    parser.add_argument("--path", type=str, default="data/brisc2025", help="Dataset directory path")
    parser.add_argument("--verify", action="store_true", help="Run verification check")
    args = parser.parse_args()

    summary = verify_dataset_integrity(args.path)
    print("=" * 60)
    print("DATASET INTEGRITY REPORT")
    print("=" * 60)
    print(f"Path: {args.path}")
    print(f"Dataset root found: {summary['dataset_found']}")
    print(f"Classification train counts: {summary['classification']['train_counts']}")
    print(f"Classification test counts: {summary['classification']['test_counts']}")
    print(f"Segmentation matched pairs: {summary['segmentation']['train_pairs']}")
    if summary["warnings"]:
        print("\nWarnings:")
        for w in summary["warnings"]:
            print(f"  [!] {w}")
    if summary["errors"]:
        print("\nErrors:")
        for e in summary["errors"]:
            print(f"  [X] {e}")
    print("=" * 60)
