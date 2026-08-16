"""
models/resnet_model.py — ResNet18 transfer-learning classifier for
MedScan AI's binary screening task.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import config


def build_model(pretrained: bool = True) -> nn.Module:
    """Loads torchvision's ResNet18 and replaces the final FC layer with a
    2-class head (benign-pattern / suspicious-pattern). The backbone starts
    fully frozen — see freeze_backbone / unfreeze_from for staged training.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(config.CLASS_NAMES)),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the final classification head.
    Used for Stage 1 head-only training."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_from(model: nn.Module, layer_name: str = config.FINETUNE_UNFREEZE_FROM) -> None:
    """Unfreeze `layer_name` onward (plus fc) for Stage 2 fine-tuning.
    Everything before `layer_name` (e.g. layer1, layer2 if unfreezing from
    layer3) stays frozen, which keeps low-level ImageNet features intact and
    only adapts higher-level features to skin-lesion imagery.
    """
    layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    if layer_name not in layer_order:
        raise ValueError(f"Unknown layer '{layer_name}', expected one of {layer_order}")
    unfreeze_from_index = layer_order.index(layer_name)
    unfreeze_set = set(layer_order[unfreeze_from_index:])

    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        param.requires_grad = top_level in unfreeze_set


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(model: nn.Module, checkpoint_path, device: torch.device) -> dict:
    """Loads model weights from a checkpoint saved by training/train.py.
    Returns the full checkpoint dict (includes metadata like epoch, val metrics)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint
