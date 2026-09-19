"""Error and confidence analysis.

Turns a classifier's predictions on a dataset into structured reports: example
selection (correct/incorrect x high/low confidence), most-confused class pairs,
cross-model disagreement, and behaviour on the saved anomalous objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mnist_validation.data.loading import DatasetBundle
from mnist_validation.models.base import Classifier, softmax
from mnist_validation.preprocessing import to_float01


@dataclass(frozen=True)
class PredictionReport:
    """Predictions and confidences of one classifier on one dataset.

    Attributes:
        labels: Ground-truth labels of shape ``(N,)``.
        predictions: Predicted labels of shape ``(N,)``.
        probabilities: Class probabilities of shape ``(N, C)``.
        confidence: Predicted-class probability of shape ``(N,)``.
        correct: Boolean mask of correct predictions.
    """

    labels: NDArray[np.int64]
    predictions: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    confidence: NDArray[np.float64]
    correct: NDArray[np.bool_]


def analyze(classifier: Classifier, bundle: DatasetBundle) -> PredictionReport:
    """Run a classifier over a bundle and return a :class:`PredictionReport`.

    Args:
        classifier: Any :class:`Classifier`.
        bundle: Dataset to evaluate.

    Returns:
        A :class:`PredictionReport`.
    """
    x = to_float01(bundle.images).astype(np.float64)
    probs = softmax(classifier.predict_logits(x))
    preds = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1)
    return PredictionReport(
        labels=bundle.labels,
        predictions=preds,
        probabilities=probs,
        confidence=confidence,
        correct=(preds == bundle.labels),
    )


def select_example_indices(report: PredictionReport, n: int = 8) -> dict[str, NDArray[np.int64]]:
    """Select example positions for the four correctness/confidence quadrants.

    Args:
        report: A prediction report.
        n: Number of examples per quadrant.

    Returns:
        Mapping with keys ``correct_high``, ``correct_low``, ``incorrect_high``,
        ``incorrect_low`` to arrays of positional indices.
    """
    correct_idx = np.flatnonzero(report.correct)
    incorrect_idx = np.flatnonzero(~report.correct)

    def _by_conf(indices: NDArray[np.int64], *, highest: bool) -> NDArray[np.int64]:
        if indices.size == 0:
            return indices
        order = np.argsort(report.confidence[indices])
        ordered = indices[order[::-1]] if highest else indices[order]
        return ordered[:n]

    return {
        "correct_high": _by_conf(correct_idx, highest=True),
        "correct_low": _by_conf(correct_idx, highest=False),
        "incorrect_high": _by_conf(incorrect_idx, highest=True),
        "incorrect_low": _by_conf(incorrect_idx, highest=False),
    }


def top_confused_pairs(confusion: NDArray[np.int64], top_n: int = 5) -> list[tuple[int, int, int]]:
    """Return the ``top_n`` most frequent (true, predicted, count) confusions.

    Args:
        confusion: Confusion matrix of shape ``(C, C)``.
        top_n: Number of pairs to return.

    Returns:
        A list of ``(true_class, predicted_class, count)`` tuples, descending by
        count, excluding the diagonal (correct predictions).
    """
    pairs: list[tuple[int, int, int]] = []
    c = confusion.shape[0]
    for i in range(c):
        for j in range(c):
            if i != j and confusion[i, j] > 0:
                pairs.append((i, j, int(confusion[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_n]


def model_disagreement_indices(reports: list[PredictionReport]) -> NDArray[np.int64]:
    """Return positions where the models' predictions are not all equal.

    Args:
        reports: Prediction reports of several models on the *same* dataset.

    Returns:
        Array of positional indices where predictions differ.
    """
    stacked = np.stack([r.predictions for r in reports], axis=0)
    disagree = (stacked != stacked[0]).any(axis=0)
    return np.flatnonzero(disagree).astype(np.int64)


def anomaly_behavior_table(
    classifiers: dict[str, Classifier], anomalies: DatasetBundle, reasons: pd.DataFrame
) -> pd.DataFrame:
    """Build a table of each model's prediction/confidence on anomalous objects.

    Args:
        classifiers: Mapping of model name to classifier.
        anomalies: Bundle of anomalous objects (their ``labels`` are the original
            labels assigned in the dataset).
        reasons: Dataframe with ``row_id`` and ``reasons`` columns.

    Returns:
        A dataframe with per-model prediction/confidence columns plus an
        ``agreement`` count of how many models match the original label.
    """
    x = to_float01(anomalies.images).astype(np.float64)
    table = pd.DataFrame(
        {
            "row_id": anomalies.row_ids,
            "original_label": anomalies.labels,
            "reasons": reasons.set_index("row_id").reindex(anomalies.row_ids)["reasons"].to_numpy(),
        }
    )
    agreement = np.zeros(len(anomalies), dtype=np.int64)
    for name, clf in classifiers.items():
        probs = softmax(clf.predict_logits(x))
        preds = probs.argmax(axis=1).astype(np.int64)
        table[f"pred_{name}"] = preds
        table[f"conf_{name}"] = probs.max(axis=1)
        agreement += (preds == anomalies.labels).astype(np.int64)
    table["models_matching_original"] = agreement
    return table
