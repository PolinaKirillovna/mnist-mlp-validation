"""Per-image statistics and class-distribution measures.

Pure functions that turn raw images and labels into tabular statistics used by
quality checks, representativeness analysis and visualisation. Nothing here has
side effects; callers decide what to persist or plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mnist_validation.config import NUM_CLASSES, NUM_PIXELS

STAT_COLUMNS = (
    "mean_intensity",
    "std_intensity",
    "nonzero_count",
    "active_fraction",
    "intensity_sum",
)


def compute_image_statistics(
    images: NDArray[np.uint8],
    labels: NDArray[np.int64],
    row_ids: NDArray[np.int64],
    active_threshold: int = 0,
) -> pd.DataFrame:
    """Compute per-image intensity/fill statistics.

    Args:
        images: Array of shape ``(N, 784)`` with pixel values in ``[0, 255]``.
        labels: Array of shape ``(N,)`` with class labels.
        row_ids: Array of shape ``(N,)`` with stable source-row identifiers.
        active_threshold: Pixel value above which a pixel counts as active.

    Returns:
        A dataframe with columns ``row_id``, ``label`` and the statistics in
        :data:`STAT_COLUMNS`.
    """
    pixels = images.astype(np.float64)
    active = images > active_threshold
    nonzero_count = active.sum(axis=1)
    frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "label": labels,
            "mean_intensity": pixels.mean(axis=1),
            "std_intensity": pixels.std(axis=1),
            "nonzero_count": nonzero_count.astype(np.int64),
            "active_fraction": nonzero_count / NUM_PIXELS,
            "intensity_sum": pixels.sum(axis=1),
        }
    )
    return frame


def class_counts(labels: NDArray[np.int64], num_classes: int = NUM_CLASSES) -> NDArray[np.int64]:
    """Return the number of samples per class.

    Args:
        labels: Array of class labels.
        num_classes: Total number of classes.

    Returns:
        Array of shape ``(num_classes,)`` with per-class counts.
    """
    return np.bincount(labels, minlength=num_classes).astype(np.int64)


def class_distribution_frame(
    labels: NDArray[np.int64], num_classes: int = NUM_CLASSES
) -> pd.DataFrame:
    """Return a dataframe of per-class counts and proportions.

    Args:
        labels: Array of class labels.
        num_classes: Total number of classes.

    Returns:
        A dataframe with columns ``class``, ``count`` and ``proportion``.
    """
    counts = class_counts(labels, num_classes)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "class": np.arange(num_classes),
            "count": counts,
            "proportion": counts / total if total else np.zeros(num_classes),
        }
    )


def imbalance_ratio(labels: NDArray[np.int64], num_classes: int = NUM_CLASSES) -> float:
    """Return the max/min class-count ratio (1.0 means perfectly balanced).

    Args:
        labels: Array of class labels.
        num_classes: Total number of classes.

    Returns:
        The ratio of the largest to the smallest class count.
    """
    counts = class_counts(labels, num_classes)
    smallest = int(counts.min())
    if smallest == 0:
        return float("inf")
    return float(counts.max() / smallest)


def distribution_entropy(labels: NDArray[np.int64], num_classes: int = NUM_CLASSES) -> float:
    """Return the Shannon entropy (bits) of the class distribution.

    A uniform distribution over ``num_classes`` classes reaches
    ``log2(num_classes)``; lower values indicate stronger imbalance.

    Args:
        labels: Array of class labels.
        num_classes: Total number of classes.

    Returns:
        Entropy of the class distribution in bits.
    """
    counts = class_counts(labels, num_classes).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def normalized_entropy(labels: NDArray[np.int64], num_classes: int = NUM_CLASSES) -> float:
    """Return class-distribution entropy normalised to ``[0, 1]``.

    Args:
        labels: Array of class labels.
        num_classes: Total number of classes.

    Returns:
        Entropy divided by ``log2(num_classes)`` (1.0 = perfectly uniform).
    """
    if num_classes <= 1:
        return 1.0
    return distribution_entropy(labels, num_classes) / float(np.log2(num_classes))
