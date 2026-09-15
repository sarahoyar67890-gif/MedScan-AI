import numpy as np
import torch
from PIL import Image

import config
from explainability.gradcam import GradCAM, overlay_heatmap_on_image
from models.resnet_model import build_model
from utils.common import get_eval_transforms


def test_gradcam_generates_valid_heatmap():
    model = build_model(pretrained=False)
    model.eval()
    gradcam = GradCAM(model, target_layer_name=config.GRADCAM_TARGET_LAYER)

    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    heatmap, predicted_class, confidence = gradcam.generate(dummy)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert predicted_class in (0, 1)
    assert 0.0 <= confidence <= 1.0


def test_overlay_matches_original_image_size():
    original = Image.new("RGB", (224, 224), color=(100, 50, 50))
    heatmap = np.random.rand(7, 7).astype("float32")

    overlay = overlay_heatmap_on_image(original, heatmap)

    assert overlay.size == original.size
    assert overlay.mode == "RGB"


def test_gradcam_respects_eval_transform_pipeline():
    model = build_model(pretrained=False)
    model.eval()
    transform = get_eval_transforms()

    image = Image.new("RGB", (400, 300), color=(80, 120, 160))
    tensor = transform(image).unsqueeze(0)

    assert tensor.shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    gradcam = GradCAM(model, target_layer_name=config.GRADCAM_TARGET_LAYER)
    heatmap, _, _ = gradcam.generate(tensor)
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0
