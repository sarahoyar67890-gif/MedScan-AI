"""
data/dataset.py — PyTorch Dataset for the HAM10000 binary screening splits
produced by data/prepare_data.py.
"""

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


class HAM10000ScreeningDataset(Dataset):
    """Loads images + binary screening labels from a split CSV
    (train.csv / val.csv / test.csv) written by prepare_data.py.
    """

    def __init__(self, csv_path: Path, transform: Optional[Callable] = None):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {csv_path}\n"
                f"Run `python -m data.prepare_data` first to generate it."
            )
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["filepath"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {image_path}: {e}")

        label = int(row["binary_label"])

        if self.transform:
            image = self.transform(image)

        return image, label

    def class_counts(self) -> dict:
        counts = self.df["binary_label"].value_counts().to_dict()
        return {
            config.CLASS_NAMES[0]: counts.get(config.LABEL_BENIGN, 0),
            config.CLASS_NAMES[1]: counts.get(config.LABEL_SUSPICIOUS, 0),
        }

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights, ordered [benign, suspicious].
        Used for the weighted CrossEntropyLoss (see training/engine.py)."""
        counts = self.df["binary_label"].value_counts().sort_index()
        n_total = counts.sum()
        n_classes = len(config.CLASS_NAMES)
        weights = n_total / (n_classes * counts)
        # Ensure both classes present even if a split is degenerate
        weights = weights.reindex([config.LABEL_BENIGN, config.LABEL_SUSPICIOUS]).fillna(1.0)
        return torch.tensor(weights.values, dtype=torch.float32)

    def compute_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler, so each training
        batch sees a roughly balanced mix of both classes despite the raw
        dataset being skewed toward benign-pattern images."""
        class_weights = self.compute_class_weights()
        labels = self.df["binary_label"].values
        sample_weights = [class_weights[label].item() for label in labels]
        return torch.tensor(sample_weights, dtype=torch.float32)
