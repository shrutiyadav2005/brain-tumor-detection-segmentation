"""Dataset and DataLoader setup for 4-class Brain MRI classification."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_IMAGE_EXTENSIONS,
    get_classification_transforms,
)

DEFAULT_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


class BrainTumorClassificationDataset:
    """Dataset manager for 4-class Brain MRI classification."""

    def __init__(
        self,
        train_dir: str,
        test_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        val_split: float = 0.15,
        seed: int = 42,
    ):
        """Initialize dataset paths and configuration.

        Args:
            train_dir: Path to directory containing class folders for training.
            test_dir: Optional path to directory containing class folders for testing.
            image_size: Target image dimensions (H, W).
            val_split: Fraction of train data reserved for validation.
            seed: Random seed for deterministic train/val split.
        """
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir) if test_dir else None
        self.image_size = image_size
        self.val_split = val_split
        self.seed = seed
        self.class_names = DEFAULT_CLASSES

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> Tuple[Optional["DataLoader"], Optional["DataLoader"], Optional["DataLoader"]]:
        """Create PyTorch DataLoaders for train, validation, and test sets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchvision are required to construct DataLoaders.")

        train_transform = get_classification_transforms(self.image_size, is_training=True)
        eval_transform = get_classification_transforms(self.image_size, is_training=False)

        train_loader, val_loader, test_loader = None, None, None

        if self.train_dir.exists():
            # Load full train directory with base eval transform initially
            full_train_dataset = datasets.ImageFolder(str(self.train_dir), transform=None)
            self.class_names = full_train_dataset.classes

            total_len = len(full_train_dataset)
            indices = list(range(total_len))

            # Deterministic split using numpy
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)

            val_size = int(np.floor(self.val_split * total_len))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]

            # Wrap subsets with distinct transforms to avoid data augmentation leakage on validation set
            class TransformWrapper(Dataset):
                def __init__(self, subset: Subset, transform):
                    self.subset = subset
                    self.transform = transform

                def __len__(self):
                    return len(self.subset)

                def __getitem__(self, idx):
                    img, label = self.subset[idx]
                    if self.transform is not None:
                        img = self.transform(img)
                    return img, label

            train_subset = Subset(full_train_dataset, train_indices)
            val_subset = Subset(full_train_dataset, val_indices)

            train_data = TransformWrapper(train_subset, train_transform)
            val_data = TransformWrapper(val_subset, eval_transform)

            train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
            )

            val_loader = DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
            )

        if self.test_dir and self.test_dir.exists():
            test_dataset = datasets.ImageFolder(str(self.test_dir), transform=eval_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
            )

        return train_loader, val_loader, test_loader
