"""Tests for classification metrics (checked against hand computations)."""

from __future__ import annotations

import numpy as np

from mnist_validation.evaluation.metrics import compute_metrics, macro_f1_from_predictions


def _one_hot_logits(preds: list[int], num_classes: int, confidence: float = 10.0) -> np.ndarray:
    logits = np.zeros((len(preds), num_classes))
    for i, p in enumerate(preds):
        logits[i, p] = confidence
    return logits


def test_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    logits = _one_hot_logits(y.tolist(), num_classes=3)
    m = compute_metrics(y, logits, num_classes=3)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.n_errors == 0
    assert m.error_rate == 0.0
    np.testing.assert_array_equal(np.diag(m.confusion), np.bincount(y, minlength=3))


def test_confusion_and_error_counts() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    preds = [0, 1, 1, 1]  # one error at index 1 (true 0, pred 1)
    logits = _one_hot_logits(preds, num_classes=2)
    m = compute_metrics(y, logits, num_classes=2)
    assert m.n_errors == 1
    assert m.error_rate == 0.25
    assert m.confusion.tolist() == [[1, 1], [0, 2]]


def test_confidence_correct_vs_incorrect() -> None:
    y = np.array([0, 1], dtype=np.int64)
    # index 0 correct with high confidence, index 1 wrong
    logits = np.array([[5.0, 0.0], [3.0, 0.0]])  # both predict class 0
    m = compute_metrics(y, logits, num_classes=2)
    # sample 0: correct; sample 1: incorrect
    assert m.mean_confidence_correct > 0.9
    assert 0.0 < m.mean_confidence_incorrect < 1.0


def test_macro_f1_helper_matches() -> None:
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    pred = np.array([0, 1, 2, 0, 1, 1], dtype=np.int64)
    logits = _one_hot_logits(pred.tolist(), num_classes=3)
    m = compute_metrics(y, logits, num_classes=3)
    assert abs(m.macro_f1 - macro_f1_from_predictions(y, pred)) < 1e-9
