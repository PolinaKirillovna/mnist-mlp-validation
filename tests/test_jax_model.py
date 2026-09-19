"""Tests for the JAX/Flax model adapter and parameter parity with PyTorch."""

from __future__ import annotations

import numpy as np

from mnist_validation.config import NUM_PIXELS
from mnist_validation.models.jax_mlp import JaxClassifier, JaxMLP, init_params
from mnist_validation.models.torch_mlp import MLP, TorchClassifier, count_parameters, resolve_device


def _jax_classifier(hidden: tuple[int, ...], activation: str = "relu") -> JaxClassifier:
    model = JaxMLP(hidden_sizes=hidden, activation=activation, dropout_p=0.0)
    params = init_params(model, seed=0, in_features=NUM_PIXELS)
    return JaxClassifier(model, params)


def test_jax_predict_shape_and_first_layer() -> None:
    clf = _jax_classifier((32, 16))
    x = np.random.default_rng(0).random((5, NUM_PIXELS))
    logits = clf.predict_logits(x)
    assert logits.shape == (5, 10)
    assert clf.first_layer_weights().shape == (32, NUM_PIXELS)
    assert clf.first_layer_preactivations(x).shape == (5, 32)


def test_jax_first_layer_preactivations_match_manual() -> None:
    clf = _jax_classifier((16,))
    x = np.random.default_rng(1).random((3, NUM_PIXELS))
    w = clf.first_layer_weights()  # (16, 784)
    kernel = np.asarray(clf.params["Dense_0"]["kernel"], dtype=np.float64)  # (784, 16)
    bias = np.asarray(clf.params["Dense_0"]["bias"], dtype=np.float64)
    np.testing.assert_allclose(clf.first_layer_preactivations(x), x @ kernel + bias, rtol=1e-4)
    np.testing.assert_allclose(w, kernel.T, rtol=1e-6)


def test_jax_and_torch_same_parameter_count() -> None:
    hidden = (256, 128)
    torch_params = count_parameters(MLP(hidden_sizes=hidden))
    jax_clf = _jax_classifier(hidden)
    jax_params = sum(
        np.asarray(v).size
        for layer in jax_clf.params.values()
        for v in layer.values()
    )
    assert torch_params == jax_params


def test_torch_and_jax_adapters_share_protocol() -> None:
    # Both adapters expose the same interface used by evaluation/interpretation.
    torch_clf = TorchClassifier(MLP(hidden_sizes=(16,)), resolve_device("cpu"))
    jax_clf = _jax_classifier((16,))
    x = np.random.default_rng(2).random((4, NUM_PIXELS))
    for clf in (torch_clf, jax_clf):
        assert clf.predict_logits(x).shape == (4, 10)
        assert clf.first_layer_weights().shape == (16, NUM_PIXELS)
