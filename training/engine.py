"""
training/engine.py — Reusable train/validate loop functions.
"""

import logging
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

log = logging.getLogger(__name__)


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    val_f1: list = field(default_factory=list)
    val_recall: list = field(default_factory=list)
    lr: list = field(default_factory=list)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> EpochResult:
    """Runs one epoch. Pass optimizer=None for evaluation-only (no backward pass)."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(dataloader.dataset)
    return EpochResult(
        loss=running_loss / n,
        accuracy=accuracy_score(all_labels, all_preds),
        precision=precision_score(all_labels, all_preds, zero_division=0),
        recall=recall_score(all_labels, all_preds, zero_division=0),
        f1=f1_score(all_labels, all_preds, zero_division=0),
    )


class EarlyStopper:
    """Stops training when validation F1 hasn't improved for `patience` epochs."""

    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Returns True if this score is the new best."""
        is_better = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if is_better:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
