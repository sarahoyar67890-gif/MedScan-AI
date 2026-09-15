"""
training/train.py — Two-stage transfer-learning training for MedScan AI.

Stage 1: train only the new classification head (backbone frozen).
Stage 2: unfreeze the deeper backbone layers and fine-tune at a lower LR.

Now supports any architecture registered in models/registry.py, not just
ResNet18 — pass --architecture to pick one. Each architecture gets its own
checkpoint files (checkpoints/medscan_<architecture>_best.pt /
_last.pt) so training one doesn't overwrite another.

Usage (from the project root):
    python -m training.train                                   # resnet18 (default)
    python -m training.train --architecture efficientnet_b0
    python -m training.train --architecture convnext_tiny
    python -m training.train --skip-finetune                    # head-only, faster on CPU
    python -m training.train --head-epochs 3 --finetune-epochs 5
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from data.dataset import HAM10000ScreeningDataset
from models import registry
from training.engine import run_epoch, EarlyStopper, TrainingHistory
from utils.common import set_seed, get_train_transforms, get_eval_transforms

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_dataloaders():
    train_ds = HAM10000ScreeningDataset(config.PROCESSED_DIR / "train.csv", transform=get_train_transforms())
    val_ds = HAM10000ScreeningDataset(config.PROCESSED_DIR / "val.csv", transform=get_eval_transforms())

    log.info("Train class distribution: %s", train_ds.class_counts())
    log.info("Val class distribution:   %s", val_ds.class_counts())

    if config.USE_WEIGHTED_SAMPLER:
        sample_weights = train_ds.compute_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
            num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds


def save_checkpoint(path: Path, model, optimizer, epoch, val_result, stage, architecture):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "architecture": architecture,
        "val_accuracy": val_result.accuracy,
        "val_precision": val_result.precision,
        "val_recall": val_result.recall,
        "val_f1": val_result.f1,
        "class_names": config.CLASS_NAMES,
    }, path)


def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 num_epochs, device, history: TrainingHistory, stage_name: str,
                 early_stopper: EarlyStopper, architecture: str,
                 best_path: Path, last_path: Path):
    for epoch in range(1, num_epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_result.f1)

        current_lr = optimizer.param_groups[0]["lr"]
        history.train_loss.append(train_result.loss)
        history.val_loss.append(val_result.loss)
        history.train_acc.append(train_result.accuracy)
        history.val_acc.append(val_result.accuracy)
        history.val_f1.append(val_result.f1)
        history.val_recall.append(val_result.recall)
        history.lr.append(current_lr)

        log.info(
            "[%s/%s] Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_recall=%.4f val_f1=%.4f | lr=%.2e",
            architecture, stage_name, epoch, num_epochs,
            train_result.loss, train_result.accuracy,
            val_result.loss, val_result.accuracy, val_result.recall, val_result.f1,
            current_lr,
        )

        is_best = early_stopper.step(val_result.f1)
        if is_best:
            save_checkpoint(best_path, model, optimizer, epoch, val_result, stage_name, architecture)
            log.info("  -> New best model saved (val_f1=%.4f) at %s", val_result.f1, best_path)

        save_checkpoint(last_path, model, optimizer, epoch, val_result, stage_name, architecture)

        if early_stopper.should_stop:
            log.info("Early stopping triggered at epoch %d (no val_f1 improvement for %d epochs).",
                      epoch, early_stopper.patience)
            break


def main():
    parser = argparse.ArgumentParser(description="Train a MedScan AI screening model")
    parser.add_argument(
        "--architecture", choices=registry.list_architectures(),
        default=config.DEFAULT_ARCHITECTURE,
        help=f"Which architecture to train. Options: {registry.list_architectures()}",
    )
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--skip-finetune", action="store_true", help="Only run Stage 1 (head-only)")
    parser.add_argument(
        "--unfreeze-from", type=str, default=None,
        help="Layer to unfreeze from for Stage 2 (defaults to a sensible per-architecture choice)",
    )
    args = parser.parse_args()
    architecture = args.architecture

    set_seed(config.SEED)
    device = config.get_device()
    log.info("Using device: %s | architecture: %s", device, architecture)

    if not (config.PROCESSED_DIR / "train.csv").exists():
        log.error(
            "No processed splits found. Run `python -m data.prepare_data` first "
            "(after placing HAM10000 files in data/raw/)."
        )
        return

    best_path, last_path = registry.checkpoint_paths(architecture, config.CHECKPOINT_DIR)

    train_loader, val_loader, train_ds = build_dataloaders()

    model = registry.build_model(architecture, pretrained=True).to(device)
    history = TrainingHistory()
    early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, mode="max")

    if config.USE_WEIGHTED_LOSS:
        class_weights = train_ds.compute_class_weights().to(device)
        log.info("Using class-weighted loss. Weights [benign, suspicious]: %s", class_weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ---------------- Stage 1: head-only ----------------
    log.info("=== [%s] Stage 1: training classification head (backbone frozen) ===", architecture)
    registry.freeze_backbone(model, architecture)
    log.info("Trainable parameters: %d", registry.count_trainable_params(model))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
    )

    train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                args.head_epochs, device, history, "stage1_head", early_stopper,
                architecture, best_path, last_path)

    # ---------------- Stage 2: fine-tune ----------------
    if not args.skip_finetune:
        unfreeze_layer = args.unfreeze_from or registry.default_unfreeze_layer(architecture)
        log.info("=== [%s] Stage 2: fine-tuning from %s onward ===", architecture, unfreeze_layer)
        registry.unfreeze_from(model, architecture, unfreeze_layer)
        log.info("Trainable parameters: %d", registry.count_trainable_params(model))

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.FINETUNE_LR, weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
        )
        early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, mode="max")

        train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                    args.finetune_epochs, device, history, "stage2_finetune", early_stopper,
                    architecture, best_path, last_path)

    # ---------------- Save training history ----------------
    history_path = config.OUTPUTS_DIR / f"training_history_{architecture}.json"
    with open(history_path, "w") as f:
        json.dump(history.__dict__, f, indent=2)
    log.info("Saved training history to %s", history_path)
    log.info("Best checkpoint: %s", best_path)
    log.info("Done. Run `python -m evaluation.evaluate --architecture %s` next for real metrics.", architecture)


if __name__ == "__main__":
    main()
