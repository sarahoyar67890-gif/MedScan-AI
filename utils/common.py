"""
utils/common.py — Shared helpers used by both training and the Streamlit app.

Keeping the transforms in one place guarantees that inference preprocessing
matches training preprocessing exactly, which matters a lot for CNNs.
"""

import random
import numpy as np
import torch
from torchvision import transforms

import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed every RNG we touch so training runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transforms() -> transforms.Compose:
    """Augmented, stochastic transforms — training only."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic transforms — used for validation, test, AND live inference
    in the Streamlit app. No randomness anywhere in this pipeline."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display purposes (e.g. Grad-CAM overlay)."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    out = tensor.cpu() * std + mean
    return out.clamp(0, 1)
