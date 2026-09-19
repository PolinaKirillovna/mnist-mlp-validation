"""Tests for splitting, undersampling and anomaly persistence (synthetic)."""

from __future__ import annotations

import numpy as np

from mnist_validation.config import NUM_PIXELS, QualityConfig
from mnist_validation.data.cleaning import (
    anomalies_frame,
    load_anomalies,
    save_anomalies,
    undersample_to_min,
)
from mnist_validation.data.loading import DatasetBundle
from mnist_validation.data.quality import collect_findings
from mnist_validation.data.splitting import (
    image_overlap_row_ids,
    stratified_train_val_split,
)


def _random_bundle(n_per_class: dict[int, int], seed: int = 0) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for cls, count in n_per_class.items():
        images.append(rng.integers(0, 256, size=(count, NUM_PIXELS), dtype=np.uint8))
        labels.append(np.full(count, cls, dtype=np.int64))
    imgs = np.concatenate(images)
    labs = np.concatenate(labels)
    return DatasetBundle(imgs, labs, np.arange(len(labs), dtype=np.int64), "syn")


def test_stratified_split_preserves_classes() -> None:
    bundle = _random_bundle({0: 100, 1: 100, 2: 100})
    train, val = stratified_train_val_split(bundle, 0.1, seed=42)
    assert len(train) == 270
    assert len(val) == 30
    # each class present in val proportionally
    assert set(val.labels.tolist()) == {0, 1, 2}


def test_undersample_balances_classes() -> None:
    bundle = _random_bundle({0: 100, 1: 40, 2: 70})
    balanced = undersample_to_min(bundle, seed=1)
    counts = np.bincount(balanced.labels, minlength=3)
    assert set(counts[:3].tolist()) == {40}


def test_image_overlap_detects_shared_images() -> None:
    shared = np.random.default_rng(0).integers(0, 256, size=(1, NUM_PIXELS), dtype=np.uint8)[0]
    train = DatasetBundle(
        np.stack([shared, np.zeros(NUM_PIXELS, np.uint8)]),
        np.array([1, 2], np.int64),
        np.array([0, 1], np.int64),
        "train",
    )
    test = DatasetBundle(
        np.stack([shared, np.full(NUM_PIXELS, 5, np.uint8)]),
        np.array([1, 3], np.int64),
        np.array([0, 1], np.int64),
        "test",
    )
    overlap = image_overlap_row_ids(train, test)
    assert overlap.tolist() == [0]


def test_anomaly_round_trip(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    # One near-empty image guarantees at least one anomaly.
    empty = np.zeros(NUM_PIXELS, dtype=np.uint8)
    rng = np.random.default_rng(3)
    normal = rng.integers(50, 200, size=(30, NUM_PIXELS), dtype=np.uint8)
    imgs = np.concatenate([empty[None, :], normal])
    labels = np.array([0] + [i % 3 for i in range(30)], dtype=np.int64)
    bundle = DatasetBundle(imgs, labels, np.arange(len(labels), dtype=np.int64), "syn")
    report = collect_findings(bundle, QualityConfig(), seed=0, pca_components=10)
    frame = anomalies_frame(bundle, report)
    assert len(frame) == report.anomaly_row_ids().size
    path = save_anomalies(frame, tmp_path / "anom.parquet")
    reloaded, meta = load_anomalies(path)
    assert reloaded.num_samples == len(frame)
    assert "reasons" in meta.columns
