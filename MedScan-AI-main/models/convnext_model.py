"""
models/convnext_model.py — ConvNeXt-Tiny transfer-learning classifier for
MedScan AI's binary screening task.

Same shape as models/resnet_model.py (a build_model function plus three
metadata constants) so models/registry.py can treat both architectures
identically. See models/registry.py for how freezing/unfreezing/
checkpointing is done generically across every architecture.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

import config

# --- Shared-interface metadata (read by models/registry.py) ---------------
HEAD_PREFIX = "classifier."
LAYER_ORDER = [
    "features.0", "features.1", "features.2", "features.3", "features.4",
    "features.5", "features.6", "features.7", "classifier",
]
GRADCAM_TARGET_LAYER = "features.7"  # final conv block before the classifier head


def build_model(pretrained: bool = True) -> nn.Module:
    """Loads torchvision's ConvNeXt-Tiny and replaces the classifier head
    with a config.CLASS_NAMES-sized head. Keeps torchvision's own LayerNorm2d
    + Flatten from the original classifier, then adds dropout + a fresh
    linear layer sized for this task. The backbone starts fully frozen —
    use models.registry.freeze_backbone / unfreeze_from for staged training.
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],  # LayerNorm2d
        model.classifier[1],  # Flatten
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(config.CLASS_NAMES)),
    )
    return model
