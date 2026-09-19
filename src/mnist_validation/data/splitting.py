"""Train/validation splitting and train-test leakage checks.

The test set is supplied separately (``mnist_test.csv``) and is the single shared
test set for every experiment and both frameworks. The training variant is split
stratified into train/validation with a fixed seed. Balancing and cleaning are
applied only to the train part; the validation set is kept as-is so that it
reflects the real (imbalanced) distribution during model selection.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from mnist_validation.data.loading import DatasetBundle

logger = logging.getLogger(__name__)


def stratified_train_val_split(
    bundle: DatasetBundle, val_fraction: float, seed: int
) -> tuple[DatasetBundle, DatasetBundle]:
    """Split a bundle into train/validation preserving class proportions.

    Args:
        bundle: Dataset to split (the training variant).
        val_fraction: Fraction assigned to the validation set.
        seed: Random seed for a reproducible split.

    Returns:
        A ``(train, val)`` tuple of :class:`DatasetBundle`.
    """
    indices = np.arange(len(bundle))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=bundle.labels,
    )
    train_idx.sort()
    val_idx.sort()
    logger.info(
        "Split '%s': %d train / %d val (val_fraction=%.2f)",
        bundle.name,
        train_idx.size,
        val_idx.size,
        val_fraction,
    )
    return (
        bundle.subset(train_idx, name=f"{bundle.name}_train"),
        bundle.subset(val_idx, name=f"{bundle.name}_val"),
    )


def _image_keys(images: NDArray[np.uint8]) -> NDArray[np.void]:
    """Return a 1-D array of per-image byte keys for exact matching."""
    arr = np.ascontiguousarray(images)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).ravel()


def image_overlap_row_ids(source: DatasetBundle, other: DatasetBundle) -> NDArray[np.int64]:
    """Return row ids in ``other`` whose image exactly matches one in ``source``.

    Used to check that the shared test set does not leak into the training data.

    Args:
        source: Reference dataset (e.g. training data).
        other: Dataset to check for overlap (e.g. test data).

    Returns:
        Row ids from ``other`` that also appear (as images) in ``source``.
    """
    source_keys = set(_image_keys(source.images).tolist())
    other_keys = _image_keys(other.images)
    mask = np.array([key in source_keys for key in other_keys.tolist()], dtype=bool)
    return cast("NDArray[np.int64]", other.row_ids[mask])
