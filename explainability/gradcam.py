"""
explainability/gradcam.py — Grad-CAM (Gradient-weighted Class Activation
Mapping) for MedScan AI's ResNet18 classifier.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).

Grad-CAM highlights which image regions most influenced the model's
prediction. It is a model-interpretability tool, not a medical explanation
— the README and the Streamlit app both make this explicit next to every
Grad-CAM visualization.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

import config


class GradCAM:
    """Hooks a target conv layer, captures its activations + gradients during
    a forward/backward pass, and produces a class-discriminative heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = config.GRADCAM_TARGET_LAYER):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        target_layer = dict(self.model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> Tuple[np.ndarray, int, float]:
        """
        input_tensor: normalized image tensor, shape (1, 3, H, W)
        target_class: class index to explain. If None, uses the model's
            predicted class (the standard Grad-CAM behavior).

        Returns: (heatmap as HxW float array in [0,1], predicted_class, confidence)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = int(torch.argmax(output, dim=1).item())
        confidence = float(probs[0, predicted_class].item())

        class_to_explain = target_class if target_class is not None else predicted_class
        score = output[0, class_to_explain]
        score.backward()

        # Global-average-pool the gradients to get per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        heatmap = F.relu(weighted_activations).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap, predicted_class, confidence


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "inferno",
) -> Image.Image:
    """Resizes the (typically 7x7) Grad-CAM heatmap up to the original image
    size and blends it on top as a colored overlay."""
    original_image = original_image.convert("RGB")
    width, height = original_image.size

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((width, height), resample=Image.BICUBIC)
    heatmap_arr = np.array(heatmap_img) / 255.0

    # matplotlib.cm.get_cmap() was deprecated in 3.7 and removed in 3.9;
    # matplotlib.colormaps[...] is the version-safe way to look up a colormap
    # by name across the >=3.7.0 range this project supports.
    colormap_fn = cm.colormaps[colormap] if hasattr(cm, "colormaps") else cm.get_cmap(colormap)
    colored_heatmap = colormap_fn(heatmap_arr)[:, :, :3]  # drop alpha channel from colormap
    colored_heatmap = np.uint8(colored_heatmap * 255)

    original_arr = np.array(original_image).astype(np.float32)
    colored_heatmap = colored_heatmap.astype(np.float32)

    blended = (1 - alpha) * original_arr + alpha * colored_heatmap
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
