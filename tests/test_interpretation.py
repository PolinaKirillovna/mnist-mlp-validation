"""Tests for occlusion and first-layer interpretation (toy linear model)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mnist_validation.config import IMAGE_SIDE, NUM_PIXELS
from mnist_validation.interpretation.first_layer import activations, weight_maps
from mnist_validation.interpretation.occlusion import (
    _accumulate_map,
    _window_positions,
    occlusion_map,
)


class ToyLinear:
    """A linear classifier implementing the Classifier protocol."""

    def __init__(self, weight: NDArray[np.float64], bias: NDArray[np.float64]) -> None:
        self.weight = weight
        self.bias = bias

    def predict_logits(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x @ self.weight.T + self.bias

    def first_layer_weights(self) -> NDArray[np.float64]:
        return self.weight

    def first_layer_preactivations(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x @ self.weight.T + self.bias


def test_window_positions_count() -> None:
    # side 4, kernel 2, stride 1 -> 3x3 = 9 positions
    assert len(_window_positions(4, 2, 1)) == 9
    # kernel 1 covers every pixel
    assert len(_window_positions(IMAGE_SIDE, 1, 1)) == NUM_PIXELS


def test_accumulate_map_averages_overlap() -> None:
    # two overlapping 2x2 windows at (0,0) and (0,1), each with drop 1.0
    m = _accumulate_map(np.array([1.0, 1.0]), [(0, 0), (0, 1)], kernel=2)
    assert m[0, 0] == 1.0  # covered once
    assert m[0, 1] == 1.0  # covered twice, (1+1)/2
    assert m[0, 2] == 1.0  # covered once (by second window)
    assert m[2, 0] == 0.0  # never covered


def test_occlusion_localises_important_pixel() -> None:
    p = 14 * IMAGE_SIDE + 14  # a central pixel
    weight = np.zeros((2, NUM_PIXELS))
    weight[1, p] = 10.0  # class 1 depends only on pixel p
    clf = ToyLinear(weight, np.zeros(2))
    image = np.zeros(NUM_PIXELS)
    image[p] = 1.0

    result = occlusion_map(clf, image, true_label=1, kernel=1, stride=1, fill=0.0)
    assert result.predicted_class == 1
    assert result.importance_pred.shape == (IMAGE_SIDE, IMAGE_SIDE)
    # the most important pixel is exactly p
    flat_argmax = int(np.argmax(result.importance_pred))
    assert flat_argmax == p
    assert result.importance_pred[14, 14] > 0.4


def test_weight_maps_shape() -> None:
    weight = np.random.default_rng(0).normal(size=(16, NUM_PIXELS))
    clf = ToyLinear(weight, np.zeros(16))
    maps = weight_maps(clf)
    assert maps.shape == (16, IMAGE_SIDE, IMAGE_SIDE)


def test_relu_activation_nonnegative() -> None:
    weight = np.random.default_rng(1).normal(size=(8, NUM_PIXELS))
    clf = ToyLinear(weight, np.zeros(8))
    x = np.random.default_rng(2).random((5, NUM_PIXELS))
    acts = activations(clf, x, "relu")
    assert acts.shape == (5, 8)
    assert (acts >= 0.0).all()
