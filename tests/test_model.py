import torch

import config
from models.resnet_model import (
    build_model,
    count_trainable_params,
    freeze_backbone,
    unfreeze_from,
)


def test_build_model_output_shape():
    model = build_model(pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, len(config.CLASS_NAMES))


def test_freeze_backbone_only_head_trainable():
    model = build_model(pretrained=False)
    freeze_backbone(model)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_unfreeze_from_layer3_keeps_early_layers_frozen():
    model = build_model(pretrained=False)
    freeze_backbone(model)
    unfreeze_from(model, "layer3")

    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        if top_level in {"layer3", "layer4", "fc"}:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should stay frozen"


def test_unfreeze_from_rejects_unknown_layer():
    model = build_model(pretrained=False)
    try:
        unfreeze_from(model, "not_a_real_layer")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_count_trainable_params_changes_with_freeze_state():
    model = build_model(pretrained=False)
    total = count_trainable_params(model)
    freeze_backbone(model)
    head_only = count_trainable_params(model)
    assert head_only < total
    assert head_only > 0
