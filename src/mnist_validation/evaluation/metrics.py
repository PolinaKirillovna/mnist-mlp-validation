"""Classification metrics.

Computed with scikit-learn and double-checked in the tests against hand-written
implementations on small arrays. Metrics operate on plain NumPy arrays so they
are framework-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from mnist_validation.config import NUM_CLASSES
from mnist_validation.models.base import softmax


@dataclass(frozen=True)
class ClassificationMetrics:
    """A bundle of classification-quality metrics.

    Attributes:
        accuracy: Overall accuracy.
        precision_per_class: Precision for each class.
        recall_per_class: Recall for each class.
        f1_per_class: F1-score for each class.
        macro_precision: Macro-averaged precision.
        macro_recall: Macro-averaged recall.
        macro_f1: Macro-averaged F1-score.
        weighted_f1: Support-weighted F1-score.
        confusion: Confusion matrix of shape ``(C, C)``.
        n_errors: Number of misclassified samples.
        error_rate: Fraction of misclassified samples.
        mean_confidence_correct: Mean predicted-class probability on correct predictions.
        mean_confidence_incorrect: Mean predicted-class probability on wrong predictions.
    """

    accuracy: float
    precision_per_class: NDArray[np.float64]
    recall_per_class: NDArray[np.float64]
    f1_per_class: NDArray[np.float64]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion: NDArray[np.int64]
    n_errors: int
    error_rate: float
    mean_confidence_correct: float
    mean_confidence_incorrect: float

    def scalar_summary(self) -> dict[str, float]:
        """Return the scalar metrics as a flat dictionary (for tables/JSON)."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "n_errors": float(self.n_errors),
            "error_rate": self.error_rate,
            "mean_confidence_correct": self.mean_confidence_correct,
            "mean_confidence_incorrect": self.mean_confidence_incorrect,
        }


def macro_f1_from_predictions(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    """Return the macro-averaged F1-score (used as the early-stopping criterion).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Macro-averaged F1-score.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_metrics(
    y_true: NDArray[np.int64],
    logits: NDArray[np.float64],
    num_classes: int = NUM_CLASSES,
) -> ClassificationMetrics:
    """Compute the full metric bundle from labels and logits.

    Args:
        y_true: True labels of shape ``(N,)``.
        logits: Model logits of shape ``(N, num_classes)``.
        num_classes: Number of classes.

    Returns:
        A populated :class:`ClassificationMetrics`.
    """
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1)
    correct = y_pred == y_true

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), average="macro", zero_division=0
    )
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    conf = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)).astype(np.int64)
    n_errors = int((~correct).sum())

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_per_class=precision.astype(np.float64),
        recall_per_class=recall.astype(np.float64),
        f1_per_class=f1.astype(np.float64),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_f1=weighted_f1,
        confusion=conf,
        n_errors=n_errors,
        error_rate=float(n_errors / len(y_true)),
        mean_confidence_correct=float(confidence[correct].mean()) if correct.any() else 0.0,
        mean_confidence_incorrect=float(confidence[~correct].mean()) if (~correct).any() else 0.0,
    )
