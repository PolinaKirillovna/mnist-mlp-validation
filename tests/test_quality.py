"""Tests for quality checks, statistics and representativeness (synthetic)."""

from __future__ import annotations

import numpy as np

from mnist_validation.config import NUM_PIXELS, QualityConfig
from mnist_validation.data.loading import DatasetBundle
from mnist_validation.data.quality import (
    find_empty_or_full,
    find_full_row_duplicates,
    find_label_conflicts,
    find_out_of_range,
    find_same_label_image_duplicates,
)
from mnist_validation.data.representativeness import (
    quantitative_representativeness,
    required_sample_size,
)
from mnist_validation.data.statistics import (
    distribution_entropy,
    imbalance_ratio,
    normalized_entropy,
)


def _bundle(images: list[np.ndarray], labels: list[int]) -> DatasetBundle:
    arr = np.stack(images).astype(np.uint8)
    lab = np.array(labels, dtype=np.int64)
    return DatasetBundle(arr, lab, np.arange(len(labels), dtype=np.int64), "synthetic")


def _img(fill: int) -> np.ndarray:
    return np.full(NUM_PIXELS, fill, dtype=np.uint8)


def _digit(active_pixels: int) -> np.ndarray:
    v = np.zeros(NUM_PIXELS, dtype=np.uint8)
    v[:active_pixels] = 200
    return v


def test_full_row_duplicates() -> None:
    a, b = _digit(100), _digit(150)
    bundle = _bundle([a, a, b], [1, 1, 2])  # rows 0 and 1 identical
    res = find_full_row_duplicates(bundle)
    assert res.flagged_row_ids.tolist() == [1]


def test_same_label_duplicate_and_conflict_distinct() -> None:
    a = _digit(100)
    # same image a with labels 1 and 2 -> conflict, not a same-label duplicate
    bundle = _bundle([a, a], [1, 2])
    conflict = find_label_conflicts(bundle)
    same_label = find_same_label_image_duplicates(bundle)
    assert conflict.flagged_row_ids.tolist() == [0, 1]
    assert same_label.flagged_row_ids.size == 0


def test_same_label_duplicate_positive() -> None:
    a = _digit(100)
    bundle = _bundle([a, a], [1, 1])
    same_label = find_same_label_image_duplicates(bundle)
    assert same_label.flagged_row_ids.tolist() == [1]


def test_out_of_range_empty_on_valid_uint8() -> None:
    bundle = _bundle([_digit(50), _digit(80)], [0, 1])
    assert find_out_of_range(bundle).flagged_row_ids.size == 0


def test_empty_and_full_detection() -> None:
    empty = _img(0)
    full = _img(255)
    normal = _digit(150)
    bundle = _bundle([empty, full, normal], [0, 1, 2])
    cfg = QualityConfig(empty_active_fraction=0.02, filled_active_fraction=0.95)
    empty_res, full_res = find_empty_or_full(bundle, cfg)
    assert empty_res.flagged_row_ids.tolist() == [0]
    assert full_res.flagged_row_ids.tolist() == [1]


def test_required_sample_size_formula() -> None:
    # t=1.96, p=0.5, delta=0.05 -> n = 1.96^2 * 0.25 / 0.0025 = 384.16
    n = required_sample_size(0.5, 1.96, 0.05)
    assert abs(n - 384.16) < 1e-6


def test_quantitative_representativeness_sufficient() -> None:
    labels = np.repeat(np.arange(10), 500).astype(np.int64)
    frame = quantitative_representativeness(labels)
    assert frame.shape[0] == 10
    assert bool(frame["sufficient"].all())


def test_imbalance_and_entropy() -> None:
    balanced = np.repeat(np.arange(10), 100).astype(np.int64)
    assert imbalance_ratio(balanced) == 1.0
    assert abs(normalized_entropy(balanced) - 1.0) < 1e-9
    assert abs(distribution_entropy(balanced) - np.log2(10)) < 1e-9

    skewed = np.array([0] * 100 + [1] * 10, dtype=np.int64)
    assert imbalance_ratio(skewed, num_classes=2) == 10.0
    assert normalized_entropy(skewed, num_classes=2) < 1.0
