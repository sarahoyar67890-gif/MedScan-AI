"""
models/registry.py — Shared model interface for MedScan AI.

Every architecture module (resnet_model.py, efficientnet_model.py,
convnext_model.py) exposes the same four things at module level:

    build_model(pretrained: bool) -> nn.Module
    HEAD_PREFIX: str            # parameter-name prefix of the classification head
    LAYER_ORDER: list[str]      # parameter-name prefixes, shallow -> deep, ending
                                 # with the head, used for staged fine-tuning
    GRADCAM_TARGET_LAYER: str   # dotted path to the last conv layer, for Grad-CAM

This file is the only place that knows how to freeze/unfreeze/checkpoint
*any* of them generically, and the only place training/evaluation code needs
to import from to get a model by name instead of hardcoding an architecture.

Nothing here trains a model or fabricates a result — it's pure architecture
plumbing so training/train.py and evaluation/evaluate.py can loop over
architectures instead of being rewritten per model.

Existing code (inference/predictor.py, app/app.py) is untouched by this file
and keeps importing models.resnet_model directly — this registry is additive,
not a replacement, until predictor.py is deliberately migrated to it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models import resnet_model, efficientnet_model, convnext_model

# Architecture name -> module implementing the shared interface documented above.
ARCHITECTURES = {
    "resnet18": resnet_model,
    "efficientnet_b0": efficientnet_model,
    "convnext_tiny": convnext_model,
}


def list_architectures() -> list[str]:
    return list(ARCHITECTURES.keys())


def get_module(architecture: str):
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Available: {list_architectures()}"
        )
    return ARCHITECTURES[architecture]


def build_model(architecture: str, pretrained: bool = True) -> nn.Module:
    return get_module(architecture).build_model(pretrained=pretrained)


def freeze_backbone(model: nn.Module, architecture: str) -> None:
    """Freeze every parameter except the classification head."""
    head_prefix = get_module(architecture).HEAD_PREFIX
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head_prefix)


def unfreeze_from(model: nn.Module, architecture: str, layer_name: str) -> None:
    """Unfreeze `layer_name` onward (plus the head) for Stage 2 fine-tuning.
    `layer_name` must be one of the architecture's LAYER_ORDER entries."""
    layer_order = get_module(architecture).LAYER_ORDER
    if layer_name not in layer_order:
        raise ValueError(
            f"Unknown layer '{layer_name}' for {architecture}, expected one of {layer_order}"
        )
    unfreeze_prefixes = tuple(layer_order[layer_order.index(layer_name):])

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(unfreeze_prefixes)


def default_unfreeze_layer(architecture: str) -> str:
    """A reasonable default 'unfreeze from here' layer for Stage 2 fine-tuning —
    roughly the last third of the network, mirroring the original resnet18
    choice of unfreezing from layer3 onward."""
    layer_order = get_module(architecture).LAYER_ORDER
    # Last entry is always the head itself; start one third of the way in
    # among the remaining (non-head) layers, minimum one layer before the head.
    backbone_layers = layer_order[:-1]
    cut = max(0, (2 * len(backbone_layers)) // 3)
    return backbone_layers[cut]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradcam_target_layer(architecture: str) -> str:
    return get_module(architecture).GRADCAM_TARGET_LAYER


def checkpoint_paths(architecture: str, checkpoint_dir: Path) -> tuple[Path, Path]:
    """Returns (best_path, last_path) for this architecture, so each model
    gets its own checkpoint files instead of overwriting a shared pair."""
    best = checkpoint_dir / f"medscan_{architecture}_best.pt"
    last = checkpoint_dir / f"medscan_{architecture}_last.pt"
    return best, last


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    """Loads model weights from a checkpoint saved by training/train.py.
    Returns the full checkpoint dict (epoch, val metrics, and which
    architecture it is, so callers can sanity-check they loaded the right one)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint
