"""
evaluation/evaluate.py — Computes real evaluation metrics on the held-out
test split from a trained checkpoint. Never fabricates numbers: if no
checkpoint exists for the requested architecture, it exits with
instructions instead of guessing.

Usage:
    python -m evaluation.evaluate                                  # resnet18 (default)
    python -m evaluation.evaluate --architecture efficientnet_b0
    python -m evaluation.evaluate --architecture convnext_tiny
"""

import argparse
import json
import logging

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
)

import config
from data.dataset import HAM10000ScreeningDataset
from models import registry
from utils.common import get_eval_transforms, set_seed

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, config.LABEL_SUSPICIOUS]
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, save_path, architecture):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(config.CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Test Set ({architecture})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(labels, probs, auc, save_path, architecture):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Test Set ({architecture})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MedScan AI checkpoint")
    parser.add_argument(
        "--architecture", choices=registry.list_architectures(),
        default=config.DEFAULT_ARCHITECTURE,
        help=f"Which architecture's checkpoint to evaluate. Options: {registry.list_architectures()}",
    )
    args = parser.parse_args()
    architecture = args.architecture

    set_seed(config.SEED)
    device = config.get_device()

    best_path, _ = registry.checkpoint_paths(architecture, config.CHECKPOINT_DIR)
    if not best_path.exists():
        log.error(
            "No trained checkpoint found at %s.\n"
            "Train it first: python -m training.train --architecture %s",
            best_path, architecture,
        )
        return

    test_csv = config.PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        log.error("No test split found. Run `python -m data.prepare_data` first.")
        return

    test_ds = HAM10000ScreeningDataset(test_csv, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    log.info("Test set class distribution: %s", test_ds.class_counts())

    model = registry.build_model(architecture, pretrained=False)
    checkpoint = registry.load_checkpoint(model, best_path, device)
    log.info("Loaded '%s' checkpoint from stage='%s', epoch=%s",
              architecture, checkpoint.get("stage"), checkpoint.get("epoch"))

    labels, preds, probs = collect_predictions(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = None  # can happen if test set only has one class present

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=config.CLASS_NAMES, zero_division=0)

    log.info("[%s] Test Accuracy:  %.4f", architecture, acc)
    log.info("[%s] Test Precision: %.4f", architecture, prec)
    log.info("[%s] Test Recall:    %.4f  (sensitivity — see README for why this matters here)", architecture, rec)
    log.info("[%s] Test F1:        %.4f", architecture, f1)
    log.info("[%s] Test ROC-AUC:   %s", architecture, f"{auc:.4f}" if auc is not None else "N/A")
    log.info("Confusion matrix:\n%s", cm)
    log.info("Classification report:\n%s", report)

    confusion_matrix_path = config.OUTPUTS_DIR / f"confusion_matrix_{architecture}.png"
    roc_curve_path = config.OUTPUTS_DIR / f"roc_curve_{architecture}.png"
    classification_report_path = config.OUTPUTS_DIR / f"classification_report_{architecture}.txt"
    metrics_json_path = config.OUTPUTS_DIR / f"metrics_{architecture}.json"

    plot_confusion_matrix(cm, confusion_matrix_path, architecture)
    if auc is not None:
        plot_roc_curve(labels, probs, auc, roc_curve_path, architecture)

    with open(classification_report_path, "w") as f:
        f.write(report)

    metrics = {
        "architecture": architecture,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_set_size": len(test_ds),
        "test_class_distribution": test_ds.class_counts(),
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Saved confusion matrix, ROC curve, classification report, and metrics.json to outputs/ "
              "(all suffixed with _%s so multiple architectures don't overwrite each other)", architecture)


if __name__ == "__main__":
    main()
