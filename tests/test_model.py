"""Tests for the PyTorch MLP and its classifier adapter (small models)."""

from __future__ import annotations

import numpy as np
import torch

from mnist_validation.models.base import predict_labels, softmax
from mnist_validation.models.torch_mlp import (
    MLP,
    TorchClassifier,
    count_parameters,
    make_activation,
    resolve_device,
)


def test_parameter_count_matches_formula() -> None:
    model = MLP(hidden_sizes=(4, 3), in_features=5, num_classes=2)
    # (5*4+4) + (4*3+3) + (3*2+2) = 24 + 15 + 8 = 47
    assert count_parameters(model) == 47


def test_forward_shape_and_logits_not_normalised() -> None:
    model = MLP(hidden_sizes=(8,), in_features=5, num_classes=3)
    out = model(torch.zeros(4, 5))
    assert out.shape == (4, 3)


def test_dropout_layers_added_only_when_positive() -> None:
    n_no_dropout = len(MLP(hidden_sizes=(4, 3), dropout_p=0.0).net)
    n_with_dropout = len(MLP(hidden_sizes=(4, 3), dropout_p=0.5).net)
    assert n_with_dropout == n_no_dropout + 2  # one Dropout per hidden layer


def test_classifier_adapter_shapes() -> None:
    model = MLP(hidden_sizes=(6, 4), in_features=5, num_classes=3)
    clf = TorchClassifier(model, resolve_device("cpu"))
    x = np.random.default_rng(0).random((7, 5))
    logits = clf.predict_logits(x)
    assert logits.shape == (7, 3)
    assert clf.first_layer_weights().shape == (6, 5)
    assert clf.first_layer_preactivations(x).shape == (7, 6)


def test_first_layer_preactivations_match_manual() -> None:
    model = MLP(hidden_sizes=(6, 4), in_features=5, num_classes=3)
    clf = TorchClassifier(model, resolve_device("cpu"))
    x = np.random.default_rng(1).random((3, 5))
    w = clf.first_layer_weights()
    linear = clf._first_linear()
    b = linear.bias.detach().cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(clf.first_layer_preactivations(x), x @ w.T + b, rtol=1e-5)


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(probs[1], [1 / 3, 1 / 3, 1 / 3], rtol=1e-6)


def test_predict_labels() -> None:
    model = MLP(hidden_sizes=(6,), in_features=5, num_classes=3)
    clf = TorchClassifier(model, resolve_device("cpu"))
    x = np.random.default_rng(2).random((4, 5))
    labels = predict_labels(clf, x)
    assert labels.shape == (4,)
    assert labels.dtype == np.int64


def test_make_activation_types() -> None:
    assert isinstance(make_activation("relu"), torch.nn.ReLU)
    assert isinstance(make_activation("gelu"), torch.nn.GELU)
