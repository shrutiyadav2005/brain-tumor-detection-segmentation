"""U-Net Architecture with ResNet34 Encoder for Tumor Segmentation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


# ==============================================================================
# FALLBACK PURE PYTORCH U-NET (when segmentation_models_pytorch is unavailable)
# ==============================================================================

class DoubleConv(nn.Module if TORCH_AVAILABLE else object):
    """(Convolution => BatchNorm => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class StandaloneUNet(nn.Module if TORCH_AVAILABLE else object):
    """Pure PyTorch U-Net architecture fallback for segmentation."""

    def __init__(self, in_channels: int = 3, classes: int = 1, base_filters: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 8, base_filters * 16))

        self.up1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_filters * 2, base_filters)

        self.outc = nn.Conv2d(base_filters, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x1], dim=1))
        logits = self.outc(x)
        return logits


# ==============================================================================
# UNIFIED SEGMENTER CLASS
# ==============================================================================

class BrainTumorSegmenter(nn.Module if TORCH_AVAILABLE else object):
    """U-Net with ResNet-34 encoder for brain tumor mask segmentation."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        """Initialize segmentation model.

        Args:
            encoder_name: Encoder backbone (e.g. 'resnet34').
            encoder_weights: Pretrained weights for encoder ('imagenet' or None).
            in_channels: Number of input channels (3 for RGB).
            classes: Number of output segmentation masks (1 for binary tumor mask).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BrainTumorSegmenter.")

        super().__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes

        if SMP_AVAILABLE:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
            self.is_smp = True
        else:
            print("[*] segmentation_models_pytorch not found. Using native PyTorch U-Net architecture.")
            self.model = StandaloneUNet(in_channels=in_channels, classes=classes)
            self.is_smp = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Args:
            x: Input tensor [batch_size, 3, H, W].

        Returns:
            Raw logits tensor [batch_size, 1, H, W].
        """
        return self.model(x)

    def predict_mask(
        self,
        x: "torch.Tensor",
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run segmentation inference on an input tensor.

        Args:
            x: Input tensor [1, 3, H, W] or [3, H, W].
            threshold: Probability threshold for binary mask.

        Returns:
            Tuple of:
                - probability_map (2D numpy float array in [0.0, 1.0])
                - binary_mask (2D numpy uint8 array with values {0, 1})
                - tumor_area_percentage (percentage of tumor pixels relative to brain)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary = (probs >= threshold).astype(np.uint8)
        tumor_pixel_count = int(np.sum(binary))
        total_pixels = binary.size
        tumor_pct = float((tumor_pixel_count / total_pixels) * 100.0)

        return probs, binary, round(tumor_pct, 2)

    def load_checkpoint(self, checkpoint_path: Union[str, Path], device: str = "cpu") -> None:
        """Load segmentation model weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        map_location = torch.device(device)
        state = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # Fallback flexible loading
            current_dict = self.state_dict()
            adapted = {}
            for k, v in state_dict.items():
                if k in current_dict and current_dict[k].shape == v.shape:
                    adapted[k] = v
                elif f"model.{k}" in current_dict and current_dict[f"model.{k}"].shape == v.shape:
                    adapted[f"model.{k}"] = v
                elif k.replace("model.", "") in current_dict and current_dict[k.replace("model.", "")].shape == v.shape:
                    adapted[k.replace("model.", "")] = v
            self.load_state_dict(adapted, strict=False)
