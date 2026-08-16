"""
training/train.py — Two-stage transfer-learning training for MedScan AI.

Stage 1: train only the new classification head (backbone frozen).
Stage 2: unfreeze the deeper backbone layers and fine-tune at a lower LR.

Usage (from the project root):
    python -m training.train
    python -m training.train --skip-finetune      # head-only, faster on CPU
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
from models.resnet_model import build_model, freeze_backbone, unfreeze_from, count_trainable_params
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


def save_checkpoint(path: Path, model, optimizer, epoch, val_result, stage):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "val_accuracy": val_result.accuracy,
        "val_precision": val_result.precision,
        "val_recall": val_result.recall,
        "val_f1": val_result.f1,
        "class_names": config.CLASS_NAMES,
    }, path)


def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 num_epochs, device, history: TrainingHistory, stage_name: str,
                 early_stopper: EarlyStopper):
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
            "[%s] Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_recall=%.4f val_f1=%.4f | lr=%.2e",
            stage_name, epoch, num_epochs,
            train_result.loss, train_result.accuracy,
            val_result.loss, val_result.accuracy, val_result.recall, val_result.f1,
            current_lr,
        )

        is_best = early_stopper.step(val_result.f1)
        if is_best:
            save_checkpoint(config.BEST_MODEL_PATH, model, optimizer, epoch, val_result, stage_name)
            log.info("  -> New best model saved (val_f1=%.4f)", val_result.f1)

        save_checkpoint(config.LAST_MODEL_PATH, model, optimizer, epoch, val_result, stage_name)

        if early_stopper.should_stop:
            log.info("Early stopping triggered at epoch %d (no val_f1 improvement for %d epochs).",
                      epoch, early_stopper.patience)
            break


def main():
    parser = argparse.ArgumentParser(description="Train MedScan AI's ResNet18 screening model")
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--skip-finetune", action="store_true", help="Only run Stage 1 (head-only)")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = config.get_device()
    log.info("Using device: %s", device)

    if not (config.PROCESSED_DIR / "train.csv").exists():
        log.error(
            "No processed splits found. Run `python -m data.prepare_data` first "
            "(after placing HAM10000 files in data/raw/)."
        )
        return

    train_loader, val_loader, train_ds = build_dataloaders()

    model = build_model(pretrained=True).to(device)
    history = TrainingHistory()
    early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, mode="max")

    if config.USE_WEIGHTED_LOSS:
        class_weights = train_ds.compute_class_weights().to(device)
        log.info("Using class-weighted loss. Weights [benign, suspicious]: %s", class_weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ---------------- Stage 1: head-only ----------------
    log.info("=== Stage 1: training classification head (backbone frozen) ===")
    freeze_backbone(model)
    log.info("Trainable parameters: %d", count_trainable_params(model))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
    )

    train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                args.head_epochs, device, history, "stage1_head", early_stopper)

    # ---------------- Stage 2: fine-tune ----------------
    if not args.skip_finetune:
        log.info("=== Stage 2: fine-tuning from %s onward ===", config.FINETUNE_UNFREEZE_FROM)
        unfreeze_from(model, config.FINETUNE_UNFREEZE_FROM)
        log.info("Trainable parameters: %d", count_trainable_params(model))

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.FINETUNE_LR, weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
        )
        early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, mode="max")

        train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                    args.finetune_epochs, device, history, "stage2_finetune", early_stopper)

    # ---------------- Save training history ----------------
    history_path = config.OUTPUTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.__dict__, f, indent=2)
    log.info("Saved training history to %s", history_path)
    log.info("Best checkpoint: %s", config.BEST_MODEL_PATH)
    log.info("Done. Run `python -m evaluation.evaluate` next to generate real metrics.")


if __name__ == "__main__":
    main()
