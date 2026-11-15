"""ResNet50-based 4-class Brain MRI Tumor Classifier."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]


class BrainTumorClassifier(nn.Module if TORCH_AVAILABLE else object):
    """ResNet50 Architecture adapted for 4-class Brain MRI tumor classification.

    Backbone: Pretrained ResNet-50
    Head: Linear projection to num_classes with optional dropout.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize the model.

        Args:
            num_classes: Number of target categories (default: 4).
            pretrained: Whether to load ImageNet pretrained weights.
            dropout_rate: Dropout probability before final classification.
            class_names: List of class name strings.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchvision are required for BrainTumorClassifier.")

        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        # Initialize ResNet50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)

        in_features = self.backbone.fc.in_features

        # Replace classification head
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through ResNet50.

        Args:
            x: Input tensor of shape [batch_size, 3, H, W].

        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        return self.backbone(x)

    def get_target_layer_for_cam(self) -> "nn.Module":
        """Return the optimal target feature layer for Grad-CAM.

        For ResNet-50, this is the final bottleneck block of layer4.
        """
        return self.backbone.layer4[-1]

    def predict(
        self,
        x: "torch.Tensor",
    ) -> Tuple[int, str, float, Dict[str, float]]:
        """Run inference on a single preprocessed input tensor.

        Args:
            x: Input tensor of shape [1, 3, H, W] or [3, H, W].

        Returns:
            Tuple of:
                - predicted_class_idx (int)
                - predicted_class_name (str)
                - confidence_score (float)
                - all_probabilities (dict mapping class_name -> probability)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        idx = int(predicted_idx.item())
        conf = float(confidence.item())
        class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"

        probs_dict = {
            self.class_names[i]: float(probabilities[i].item())
            for i in range(len(self.class_names))
        }

        return idx, class_name, conf, probs_dict

    def load_checkpoint(self, checkpoint_path: Union[str, Path], device: str = "cpu") -> None:
        """Load model weights from a state dict or checkpoint dictionary.

        Args:
            checkpoint_path: Path to .pth file.
            device: Target device for loading.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        map_location = torch.device(device)
        state = torch.load(checkpoint_path, map_location=map_location)

        # Handle both checkpoint dicts and pure state dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state

        # If saved model had a simple Linear fc and self has Sequential fc, or vice versa, adapt
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            # Attempt flexible key loading
            current_dict = self.state_dict()
            adapted_dict = {}
            for k, v in state_dict.items():
                if k in current_dict and current_dict[k].shape == v.shape:
                    adapted_dict[k] = v
                elif k.replace("backbone.", "") in current_dict and current_dict[k.replace("backbone.", "")].shape == v.shape:
                    adapted_dict[k.replace("backbone.", "")] = v
                elif f"backbone.{k}" in current_dict and current_dict[f"backbone.{k}"].shape == v.shape:
                    adapted_dict[f"backbone.{k}"] = v
            self.load_state_dict(adapted_dict, strict=False)
