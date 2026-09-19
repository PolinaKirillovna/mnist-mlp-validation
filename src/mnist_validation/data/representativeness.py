"""Sample representativeness analysis.

Quantitative representativeness follows the methodichka formula

    n = t^2 * p * q / Δ^2,

with ``t = 1.96`` (95% confidence) and ``Δ = 0.05`` (allowed margin of error);
``p`` is a class proportion and ``q = 1 - p``. Qualitative representativeness is
summarised by the class-imbalance ratio and the (normalised) distribution entropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mnist_validation.config import NUM_CLASSES
from mnist_validation.data.statistics import (
    class_counts,
    imbalance_ratio,
    normalized_entropy,
)

DEFAULT_T_VALUE = 1.96
DEFAULT_MARGIN = 0.05


def required_sample_size(proportion: float, t_value: float, margin: float) -> float:
    """Return the required sample size for a proportion estimate.

    Args:
        proportion: Feature proportion ``p`` in ``[0, 1]``.
        t_value: Confidence coefficient ``t``.
        margin: Allowed margin of error ``Δ``.

    Returns:
        The required number of observations ``n``.
    """
    q = 1.0 - proportion
    return (t_value**2) * proportion * q / (margin**2)


def quantitative_representativeness(
    labels: NDArray[np.int64],
    t_value: float = DEFAULT_T_VALUE,
    margin: float = DEFAULT_MARGIN,
    num_classes: int = NUM_CLASSES,
) -> pd.DataFrame:
    """Compare the actual per-class counts with the required sample size.

    Args:
        labels: Array of class labels.
        t_value: Confidence coefficient ``t``.
        margin: Allowed margin of error ``Δ``.
        num_classes: Total number of classes.

    Returns:
        A dataframe with columns ``class``, ``count``, ``proportion``,
        ``required_n`` and ``sufficient``.
    """
    counts = class_counts(labels, num_classes)
    total = int(counts.sum())
    rows = []
    for cls in range(num_classes):
        proportion = counts[cls] / total if total else 0.0
        req = required_sample_size(proportion, t_value, margin)
        rows.append(
            {
                "class": cls,
                "count": int(counts[cls]),
                "proportion": proportion,
                "required_n": req,
                "sufficient": bool(counts[cls] >= req),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class QualitativeRepresentativeness:
    """Qualitative representativeness summary.

    Attributes:
        imbalance_ratio: Ratio of the largest to the smallest class count.
        entropy_normalized: Class-distribution entropy normalised to ``[0, 1]``.
        min_class: Class index with the fewest samples.
        max_class: Class index with the most samples.
    """

    imbalance_ratio: float
    entropy_normalized: float
    min_class: int
    max_class: int


def qualitative_representativeness(
    labels: NDArray[np.int64], num_classes: int = NUM_CLASSES
) -> QualitativeRepresentativeness:
    """Summarise how close the class distribution is to uniform.

    Args:
        labels: Array of class labels.
        num_classes: Total number of classes.

    Returns:
        A :class:`QualitativeRepresentativeness` summary.
    """
    counts = class_counts(labels, num_classes)
    return QualitativeRepresentativeness(
        imbalance_ratio=imbalance_ratio(labels, num_classes),
        entropy_normalized=normalized_entropy(labels, num_classes),
        min_class=int(np.argmin(counts)),
        max_class=int(np.argmax(counts)),
    )
