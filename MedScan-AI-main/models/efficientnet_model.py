"""
models/efficientnet_model.py — EfficientNet-B0 transfer-learning classifier
for MedScan AI's binary screening task.

Same shape as models/resnet_model.py (a build_model function plus three
metadata constants) so models/registry.py can treat both architectures
identically. See models/resnet_model.py for the original binary-screening
architecture this mirrors, and models/registry.py for how freezing/
unfreezing/checkpointing is done generically across every architecture.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

import config

# --- Shared-interface metadata (read by models/registry.py) ---------------
HEAD_PREFIX = "classifier."
LAYER_ORDER = [
    "features.0", "features.1", "features.2", "features.3", "features.4",
    "features.5", "features.6", "features.7", "features.8", "classifier",
]
GRADCAM_TARGET_LAYER = "features.8"  # final conv block before global pooling


def build_model(pretrained: bool = True) -> nn.Module:
    """Loads torchvision's EfficientNet-B0 and replaces the classifier head
    with a config.CLASS_NAMES-sized head. The backbone starts fully frozen —
    use models.registry.freeze_backbone / unfreeze_from for staged training.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(config.CLASS_NAMES)),
    )
    return model
