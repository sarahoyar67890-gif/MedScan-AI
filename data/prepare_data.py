"""
data/prepare_data.py — Turn the raw HAM10000 download into reproducible,
leakage-free train/val/test CSV splits.

WHY LESION-LEVEL SPLITTING MATTERS
-----------------------------------
HAM10000 contains multiple images of the SAME physical lesion (same
`lesion_id`) taken at different angles/zooms. If two images of the same
lesion end up in different splits (e.g. one in train, one in test), the
model can partly "recognize" that lesion rather than generalize — this is
data leakage and it inflates test performance. This script groups by
`lesion_id` first, then splits, so every image of a given lesion stays in
exactly one split.

EXPECTED INPUT LAYOUT (place these under data/raw/, unmodified from Kaggle
or the ISIC archive download):

    data/raw/HAM10000_metadata.csv
    data/raw/HAM10000_images_part_1/*.jpg
    data/raw/HAM10000_images_part_2/*.jpg

(If your download already merged both image folders into one, that's fine
too — the script searches every subfolder under data/raw/ for the image.)

USAGE
-----
    python -m data.prepare_data
"""

import sys
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _find_image_path(image_id: str) -> Path | None:
    """Search data/raw/**/ for <image_id>.jpg. Returns None if not found."""
    candidates = list(config.RAW_DATA_DIR.rglob(f"{image_id}.jpg"))
    if not candidates:
        candidates = list(config.RAW_DATA_DIR.rglob(f"{image_id}.png"))
    return candidates[0] if candidates else None


def build_splits() -> pd.DataFrame:
    metadata_path = config.RAW_DATA_DIR / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        log.error(
            "Could not find %s.\n\n"
            "Download HAM10000 first (see README 'Dataset Setup') and place:\n"
            "  data/raw/HAM10000_metadata.csv\n"
            "  data/raw/HAM10000_images_part_1/  (and part_2)\n",
            metadata_path,
        )
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    required_cols = {"image_id", "lesion_id", "dx"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error("Metadata CSV is missing expected columns: %s", missing)
        sys.exit(1)

    log.info("Loaded metadata: %d rows, %d unique lesions", len(df), df["lesion_id"].nunique())

    # --- Map to binary screening label, skipping any unrecognized dx codes explicitly ---
    def safe_map(dx: str):
        try:
            return config.dx_to_binary_label(dx)
        except ValueError as e:
            log.warning("%s — dropping row", e)
            return None

    df["binary_label"] = df["dx"].apply(safe_map)
    n_before = len(df)
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)
    if len(df) != n_before:
        log.warning("Dropped %d rows with unmapped dx codes", n_before - len(df))

    # --- Resolve actual image file paths, drop rows whose image is missing ---
    log.info("Resolving image file paths under %s ...", config.RAW_DATA_DIR)
    df["filepath"] = df["image_id"].apply(_find_image_path)
    missing_images = df["filepath"].isna().sum()
    if missing_images:
        log.warning(
            "%d rows reference images that were not found under data/raw/. "
            "These rows will be excluded. Check your download is complete.",
            missing_images,
        )
    df = df.dropna(subset=["filepath"]).copy()
    df["filepath"] = df["filepath"].astype(str)

    if df.empty:
        log.error("No usable rows remain after resolving images. Check your data/raw/ layout.")
        sys.exit(1)

    # --- Class distribution report (before splitting) ---
    dist = df["dx"].value_counts()
    binary_dist = df["binary_label"].value_counts().rename(index={0: "benign-pattern", 1: "suspicious-pattern"})
    log.info("Original 7-class distribution:\n%s", dist.to_string())
    log.info("Binary screening distribution:\n%s", binary_dist.to_string())
    imbalance_ratio = binary_dist.max() / binary_dist.min()
    log.info("Class imbalance ratio (majority:minority) ≈ %.2f : 1", imbalance_ratio)

    # --- Lesion-grouped split: test set first, then val from the remainder ---
    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SPLIT, random_state=config.SEED)
    trainval_idx, test_idx = next(gss1.split(df, groups=df["lesion_id"]))
    trainval_df = df.iloc[trainval_idx]
    test_df = df.iloc[test_idx]

    val_fraction_of_trainval = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_trainval, random_state=config.SEED)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df["lesion_id"]))
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]

    # --- Sanity check: confirm zero lesion_id overlap across splits ---
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    assert not (train_lesions & val_lesions), "Leakage detected: train/val lesion overlap!"
    assert not (train_lesions & test_lesions), "Leakage detected: train/test lesion overlap!"
    assert not (val_lesions & test_lesions), "Leakage detected: val/test lesion overlap!"

    log.info(
        "Split sizes (images) — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    log.info(
        "Split sizes (unique lesions) — train: %d | val: %d | test: %d",
        len(train_lesions), len(val_lesions), len(test_lesions),
    )

    keep_cols = ["image_id", "lesion_id", "dx", "binary_label", "filepath"]
    train_df[keep_cols].to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    val_df[keep_cols].to_csv(config.PROCESSED_DIR / "val.csv", index=False)
    test_df[keep_cols].to_csv(config.PROCESSED_DIR / "test.csv", index=False)

    log.info("Wrote train.csv / val.csv / test.csv to %s", config.PROCESSED_DIR)
    return df


if __name__ == "__main__":
    build_splits()
