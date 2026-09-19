"""Tests for dataset loading and format validation (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnist_validation.config import NUM_PIXELS
from mnist_validation.data.loading import DatasetBundle, load_bundle, validate_frame


def _synthetic_frame(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(n, NUM_PIXELS), dtype=np.uint8)
    cols = [f"{r}x{c}" for r in range(1, 29) for c in range(1, 29)]
    frame = pd.DataFrame(pixels, columns=cols)
    frame.insert(0, "label", rng.integers(0, 10, size=n))
    return frame


def test_validate_frame_accepts_valid_frame() -> None:
    validate_frame(_synthetic_frame(), "synthetic")


def test_validate_frame_rejects_missing_label() -> None:
    frame = _synthetic_frame().drop(columns=["label"])
    with pytest.raises(ValueError, match="label"):
        validate_frame(frame, "synthetic")


def test_validate_frame_rejects_wrong_pixel_count() -> None:
    frame = _synthetic_frame().drop(columns=["1x1"])
    with pytest.raises(ValueError, match="pixel columns"):
        validate_frame(frame, "synthetic")


def test_validate_frame_rejects_nan() -> None:
    frame = _synthetic_frame().astype(float)
    frame.loc[0, "1x1"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_frame(frame, "synthetic")


def test_load_bundle_round_trip(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    frame = _synthetic_frame(7)
    path = tmp_path / "mini.csv"
    frame.to_csv(path, index=False)
    bundle = load_bundle(path, "mini")
    assert bundle.num_samples == 7
    assert bundle.images.shape == (7, NUM_PIXELS)
    assert bundle.images.dtype == np.uint8
    np.testing.assert_array_equal(bundle.labels, frame["label"].to_numpy())


def test_bundle_subset_and_reshape() -> None:
    images = np.zeros((4, NUM_PIXELS), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    row_ids = np.arange(4, dtype=np.int64)
    bundle = DatasetBundle(images=images, labels=labels, row_ids=row_ids, name="t")
    assert bundle.images_2d().shape == (4, 28, 28)
    sub = bundle.subset(np.array([1, 3]))
    assert sub.num_samples == 2
    np.testing.assert_array_equal(sub.labels, [1, 3])


def test_bundle_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="784"):
        DatasetBundle(
            images=np.zeros((2, 10), dtype=np.uint8),
            labels=np.zeros(2, dtype=np.int64),
            row_ids=np.zeros(2, dtype=np.int64),
            name="bad",
        )
