"""
config.py — Central configuration for MedScan AI.

All paths use pathlib so the project runs unmodified on Windows, macOS, and Linux.
Nothing in this file trains a model or touches the network — it is pure configuration.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"                # put HAM10000_metadata.csv + image folders here
PROCESSED_DIR = DATA_DIR / "processed"         # generated train/val/test split CSVs land here

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "medscan_resnet18_best.pt"
LAST_MODEL_PATH = CHECKPOINT_DIR / "medscan_resnet18_last.pt"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HISTORY_PLOT_PATH = OUTPUTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = OUTPUTS_DIR / "roc_curve.png"
METRICS_JSON_PATH = OUTPUTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / "classification_report.txt"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HAM10000 -> screening-category mapping
#
# HAM10000's original 7 diagnostic classes ("dx" column):
#   akiec  Actinic keratoses / intraepithelial carcinoma (pre-malignant / early malignant)
#   bcc    Basal cell carcinoma (malignant)
#   bkl    Benign keratosis-like lesions
#   df     Dermatofibroma
#   mel    Melanoma (malignant)
#   nv     Melanocytic nevi (benign mole)
#   vasc   Vascular lesions
#
# MedScan AI is a BINARY screening experiment, not 7-way diagnosis. The mapping
# below groups malignant/pre-malignant categories into "suspicious" and the
# remaining categories into "benign-pattern". This is a simplification made
# explicit here — akiec is borderline (pre-malignant/early malignant) and is
# grouped with the "suspicious" side deliberately, erring toward higher recall
# on the side that would prompt a person to seek real medical review.
# This mapping choice is a modeling decision, not a clinical one, and is
# documented in the README so nobody mistakes it for a diagnostic standard.
# ---------------------------------------------------------------------------
DX_FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

SUSPICIOUS_CLASSES = {"akiec", "bcc", "mel"}
BENIGN_CLASSES = {"bkl", "df", "nv", "vasc"}

# Binary label indices used everywhere in the codebase.
LABEL_BENIGN = 0
LABEL_SUSPICIOUS = 1
CLASS_NAMES = ["benign-pattern", "suspicious-pattern"]  # index 0, index 1


def dx_to_binary_label(dx: str) -> int:
    """Map a HAM10000 'dx' code to the binary screening label. Raises on unknown codes
    instead of silently defaulting, so mapping mistakes surface immediately."""
    dx = dx.strip().lower()
    if dx in SUSPICIOUS_CLASSES:
        return LABEL_SUSPICIOUS
    if dx in BENIGN_CLASSES:
        return LABEL_BENIGN
    raise ValueError(f"Unrecognized HAM10000 dx code: '{dx}'")


# ---------------------------------------------------------------------------
# Image / model settings
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224  # standard ResNet18 input resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The final conv block used for Grad-CAM on torchvision's resnet18.
GRADCAM_TARGET_LAYER = "layer4"

# ---------------------------------------------------------------------------
# Training hyperparameters (safe CPU-trainable defaults; edit freely)
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2

# Stage 1: train only the new classification head (backbone frozen)
HEAD_EPOCHS = 5
HEAD_LR = 1e-3

# Stage 2: optional fine-tuning of the deeper backbone layers at a lower LR
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4
FINETUNE_UNFREEZE_FROM = "layer3"  # unfreeze layer3 + layer4 + fc

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15  # remaining ~0.70 goes to train

# Class imbalance strategy: weighted random sampling during training +
# class-weighted loss. Both are implemented in training/engine.py; see the
# README section "Class Imbalance" for why both are used together.
USE_WEIGHTED_SAMPLER = True
USE_WEIGHTED_LOSS = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
