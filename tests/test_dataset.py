import pandas as pd
import pytest

import config
from data.dataset import HAM10000ScreeningDataset


@pytest.fixture
def imbalanced_split_csv(tmp_path):
    # 8 benign, 2 suspicious — mimics HAM10000's real skew, at toy scale.
    rows = []
    for i in range(8):
        rows.append({"filepath": f"/fake/benign_{i}.jpg", "binary_label": config.LABEL_BENIGN})
    for i in range(2):
        rows.append({"filepath": f"/fake/suspicious_{i}.jpg", "binary_label": config.LABEL_SUSPICIOUS})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "toy_split.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_missing_split_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HAM10000ScreeningDataset(tmp_path / "does_not_exist.csv")


def test_class_counts(imbalanced_split_csv):
    ds = HAM10000ScreeningDataset(imbalanced_split_csv)
    counts = ds.class_counts()
    assert counts[config.CLASS_NAMES[0]] == 8
    assert counts[config.CLASS_NAMES[1]] == 2


def test_class_weights_favor_minority_class(imbalanced_split_csv):
    ds = HAM10000ScreeningDataset(imbalanced_split_csv)
    weights = ds.compute_class_weights()
    # Minority class (suspicious, index 1) should get a larger weight.
    assert weights[config.LABEL_SUSPICIOUS] > weights[config.LABEL_BENIGN]


def test_sample_weights_length_matches_dataset(imbalanced_split_csv):
    ds = HAM10000ScreeningDataset(imbalanced_split_csv)
    sample_weights = ds.compute_sample_weights()
    assert len(sample_weights) == len(ds)


def test_len_matches_row_count(imbalanced_split_csv):
    ds = HAM10000ScreeningDataset(imbalanced_split_csv)
    assert len(ds) == 10
