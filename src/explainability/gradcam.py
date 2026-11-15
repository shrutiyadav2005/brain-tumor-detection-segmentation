"""Explainable AI: Grad-CAM Implementation for ResNet50 Brain Tumor Classifier."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union
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

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NativeGradCAM:
    """Pure PyTorch implementation of Gradient-weighted Class Activation Mapping (Grad-CAM).

    Extracts gradients from the target convolutional layer during backprop to compute
    channel importance weights, producing a localized class activation heatmap without
    requiring external libraries.
    """

    def __init__(self, model: "torch.nn.Module", target_layer: "torch.nn.Module"):
        """Initialize Grad-CAM hooks on target model layer.

        Args:
            model: PyTorch model (in eval mode).
            target_layer: The target convolutional layer (e.g. model.backbone.layer4[-1]).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handlers = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handlers.extend([h1, h2])

    def generate_cam(
        self,
        input_tensor: "torch.Tensor",
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM activation heatmap for a given input tensor.

        Args:
            input_tensor: Preprocessed tensor of shape [1, 3, H, W].
            target_class_idx: Target class index. If None, uses top predicted class.

        Returns:
            Normalized 2D heatmap array of shape (H, W) with values in [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        score.backward()

        # Global average pooling of gradients: weights alpha_k
        gradients = self.gradients[0]  # Shape: [C, H_feat, W_feat]
        activations = self.activations[0]  # Shape: [C, H_feat, W_feat]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # Shape: [C, 1, 1]
        cam = torch.sum(weights * activations, dim=0)  # Shape: [H_feat, W_feat]

        # ReLU on combined activations (we only care about positive influences)
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Upsample heatmap to match input tensor spatial resolution (H, W)
        _, _, h, w = input_tensor.shape
        cam_img = Image.fromarray((cam_norm * 255).astype(np.uint8)).resize(
            (w, h), resample=Image.Resampling.BILINEAR
        )
        cam_upsampled = np.array(cam_img, dtype=np.float32) / 255.0
        return cam_upsampled

    def remove_hooks(self) -> None:
        """Clean up forward and backward hooks."""
        for h in self._handlers:
            h.remove()
        self._handlers.clear()


class GradCAMExplainer:
    """High-level interface for generating and visualizing Grad-CAM overlays."""

    def __init__(self, classifier_model: "torch.nn.Module", target_layer: Optional["torch.nn.Module"] = None):
        """Initialize explainer.

        Args:
            classifier_model: Trained BrainTumorClassifier instance.
            target_layer: Optional target layer. Defaults to classifier_model.get_target_layer_for_cam().
        """
        self.model = classifier_model
        self.target_layer = target_layer or (
            self.model.get_target_layer_for_cam() if hasattr(self.model, "get_target_layer_for_cam") else None
        )
        self.grad_cam = NativeGradCAM(self.model, self.target_layer) if (TORCH_AVAILABLE and self.target_layer) else None

    def explain(
        self,
        input_tensor: "torch.Tensor",
        original_image: Union[np.ndarray, Image.Image],
        target_class_idx: Optional[int] = None,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM heatmap and blended visualization.

        Args:
            input_tensor: Preprocessed tensor [1, 3, H, W].
            original_image: Original MRI image (PIL or numpy).
            target_class_idx: Target class index.
            alpha: Opacity factor for heatmap overlay (0 = only image, 1 = only heatmap).
            colormap: Matplotlib colormap name ('jet', 'inferno', 'viridis').

        Returns:
            Tuple of:
                - 2D normalized heatmap [0, 1]
                - RGB blended overlay image [0, 1]
        """
        import matplotlib.pyplot as plt

        if isinstance(original_image, Image.Image):
            orig_np = np.array(original_image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            orig_np = np.array(original_image, dtype=np.float32)
            if orig_np.max() > 1.0:
                orig_np /= 255.0
            if orig_np.ndim == 2:
                orig_np = np.stack([orig_np] * 3, axis=-1)

        if self.grad_cam is None:
            # Fallback mock heatmap for test environments without torch
            h, w = orig_np.shape[:2]
            dummy_heatmap = np.zeros((h, w), dtype=np.float32)
            return dummy_heatmap, orig_np

        heatmap = self.grad_cam.generate_cam(input_tensor, target_class_idx=target_class_idx)

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored_heatmap = cmap(heatmap)[..., :3]  # Strip alpha channel

        # Resize heatmap if shape mismatch
        if colored_heatmap.shape[:2] != orig_np.shape[:2]:
            h, w = orig_np.shape[:2]
            cmap_pil = Image.fromarray((colored_heatmap * 255).astype(np.uint8)).resize((w, h))
            colored_heatmap = np.array(cmap_pil, dtype=np.float32) / 255.0

        # Blend
        overlay = (1.0 - alpha) * orig_np + alpha * colored_heatmap
        overlay = np.clip(overlay, 0.0, 1.0)

        return heatmap, overlay

    @staticmethod
    def get_clinical_interpretation_notes() -> Dict[str, str]:
        """Return clinical context and limitations of Grad-CAM for medical AI."""
        return {
            "what_it_is": (
                "Grad-CAM (Gradient-weighted Class Activation Mapping) visually highlights which "
                "spatial regions in the MRI scan most strongly drove the ResNet50 neural network "
                "toward its predicted tumor classification."
            ),
            "how_it_works": (
                "It pools the backpropagated gradients at the final convolutional layer (layer4) "
                "to assign importance weights to each of the 2,048 feature activation maps."
            ),
            "clinical_limitations": (
                "Grad-CAM provides COARSE, low-resolution visual attribution (7x7 pixels upsampled). "
                "It frequently highlights secondary contextual indicators (e.g. surrounding peritumoral edema, "
                "ventricular compression, or midline shift). It CANNOT and MUST NOT be interpreted as a "
                "precise anatomical tumor boundary or surgical resection margin. Spatial tumor boundaries "
                "are computed separately by the U-Net segmentation network."
            ),
        }
