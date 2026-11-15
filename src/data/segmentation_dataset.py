"""Dataset and DataLoader setup for pixel-level Brain Tumor Segmentation."""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_IMAGE_EXTENSIONS,
    preprocess_mask,
)


class BrainTumorSegmentationDataset(Dataset):
    """Pairwise dataset for Brain MRI images and corresponding binary tumor masks."""

    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
    ):
        """Initialize dataset with verified pairwise matching.

        Args:
            images_dir: Directory containing MRI scan images.
            masks_dir: Directory containing ground-truth tumor masks.
            image_size: Target height and width (H, W).
            is_training: Flag to apply training data augmentations.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.is_training = is_training

        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []

        if self.images_dir.exists() and self.masks_dir.exists():
            self._pair_files()

    def _pair_files(self) -> None:
        """Pair images and masks by file stem to avoid arbitrary sorting misalignment."""
        all_imgs = {
            f.stem: f for f in self.images_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not f.name.startswith(".")
        }
        all_msks = {
            f.stem: f for f in self.masks_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not f.name.startswith(".")
        }

        # Check for matching stems
        common_stems = sorted(list(set(all_imgs.keys()).intersection(set(all_msks.keys()))))
        if not common_stems:
            # Fallback: check if mask has a suffix like '_mask'
            common_stems = []
            for stem, img_path in sorted(all_imgs.items()):
                mask_candidate = self.masks_dir / f"{stem}_mask.png"
                if not mask_candidate.exists():
                    mask_candidate = self.masks_dir / f"{stem}_mask.jpg"
                if mask_candidate.exists():
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_candidate)

            if not self.image_paths:
                # If neither matched directly, warn and pair by sorted order if counts match
                if len(all_imgs) == len(all_msks) and len(all_imgs) > 0:
                    sorted_imgs = sorted(all_imgs.values())
                    sorted_msks = sorted(all_msks.values())
                    self.image_paths = sorted_imgs
                    self.mask_paths = sorted_msks
        else:
            for stem in common_stems:
                self.image_paths.append(all_imgs[stem])
                self.mask_paths.append(all_msks[stem])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Retrieve paired MRI scan and binary mask tensor.

        Returns:
            Tuple of:
                - Image tensor of shape [3, H, W] normalized with ImageNet stats.
                - Mask tensor of shape [1, H, W] with values in {0.0, 1.0}.
        """
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]

        # 1. Load image and convert to RGB
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = pil_img.resize(self.image_size, resample=Image.Resampling.BILINEAR)

        # 2. Load mask and enforce NEAREST NEIGHBOR interpolation & binarization
        mask_binary = preprocess_mask(msk_path, target_size=self.image_size)

        if not TORCH_AVAILABLE:
            return np.array(pil_img), mask_binary

        # 3. Convert image to PyTorch tensor and normalize
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        # Transpose from (H, W, C) to (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Normalize with ImageNet mean and std
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Mask tensor: add channel dimension [1, H, W]
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return img_tensor, mask_tensor


def get_segmentation_dataloaders(
    train_images: str,
    train_masks: str,
    test_images: Optional[str] = None,
    test_masks: Optional[str] = None,
    image_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[Optional["DataLoader"], Optional["DataLoader"], Optional["DataLoader"]]:
    """Create train, validation, and test DataLoaders for segmentation.

    If dedicated test directories are not provided, deterministically partitions
    train data into train, val, and test splits.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to construct DataLoaders.")

    train_dir_img = Path(train_images)
    train_dir_msk = Path(train_masks)

    if not (train_dir_img.exists() and train_dir_msk.exists()):
        return None, None, None

    full_dataset = BrainTumorSegmentationDataset(
        images_dir=train_dir_img,
        masks_dir=train_dir_msk,
        image_size=image_size,
    )

    total_len = len(full_dataset)
    if total_len == 0:
        return None, None, None

    # Dedicated test set available?
    has_test_dir = test_images and test_masks and Path(test_images).exists() and Path(test_masks).exists()

    indices = list(range(total_len))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    if has_test_dir:
        val_size = int(np.floor(val_split * total_len))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = BrainTumorSegmentationDataset(test_images, test_masks, image_size=image_size)
    else:
        # Partition single dataset into train / val / test
        val_size = int(np.floor(val_split * total_len))
        test_size = int(np.floor(test_split * total_len))

        val_indices = indices[:val_size]
        test_indices = indices[val_size:val_size + test_size]
        train_indices = indices[val_size + test_size:]

        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
