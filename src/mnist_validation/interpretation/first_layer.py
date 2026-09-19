"""First-layer weight and activation interpretation.

Each neuron of the first fully-connected layer has a 784-dimensional weight
vector that can be reshaped to ``28 x 28`` and shown as an image: positive
weights mark input regions that increase the neuron's pre-activation, negative
weights mark regions that decrease it. Because the input is scaled to ``[0, 1]``
(non-negative), an active pixel over a positive-weight region raises the response
and over a negative-weight region lowers it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mnist_validation.config import IMAGE_SIDE, NUM_CLASSES
from mnist_validation.models.base import Classifier

ActivationFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_ACTIVATION_FNS: dict[str, ActivationFn] = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "gelu": lambda z: 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3))),
}


def activation_function(name: str) -> ActivationFn:
    """Return a NumPy activation function by name (relu/tanh/gelu)."""
    if name not in _ACTIVATION_FNS:
        raise KeyError(f"Unknown activation '{name}'")
    return _ACTIVATION_FNS[name]


def weight_maps(classifier: Classifier) -> NDArray[np.float64]:
    """Return first-layer weights reshaped to ``(H, 28, 28)``.

    Args:
        classifier: Any :class:`Classifier`.

    Returns:
        Array of shape ``(H, 28, 28)``.
    """
    weights = classifier.first_layer_weights()
    return weights.reshape(weights.shape[0], IMAGE_SIDE, IMAGE_SIDE)


def preactivations(classifier: Classifier, images: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return first-layer pre-activations of shape ``(N, H)``.

    Args:
        classifier: Any :class:`Classifier`.
        images: Normalised images of shape ``(N, 784)``.

    Returns:
        Array of shape ``(N, H)``.
    """
    return classifier.first_layer_preactivations(images)


def activations(
    classifier: Classifier, images: NDArray[np.float64], activation: str
) -> NDArray[np.float64]:
    """Return first-layer post-activations of shape ``(N, H)``.

    Args:
        classifier: Any :class:`Classifier`.
        images: Normalised images of shape ``(N, 784)``.
        activation: Activation name matching the trained model.

    Returns:
        Array of shape ``(N, H)``.
    """
    return activation_function(activation)(preactivations(classifier, images))


def mean_activation_by_class(
    classifier: Classifier,
    images: NDArray[np.float64],
    labels: NDArray[np.int64],
    activation: str,
    num_classes: int = NUM_CLASSES,
) -> NDArray[np.float64]:
    """Return an ``(H, num_classes)`` heatmap of mean neuron activation per class.

    Args:
        classifier: Any :class:`Classifier`.
        images: Normalised images of shape ``(N, 784)``.
        labels: Class labels of shape ``(N,)``.
        activation: Activation name matching the trained model.
        num_classes: Number of classes.

    Returns:
        Array of shape ``(H, num_classes)``.
    """
    acts = activations(classifier, images, activation)
    hidden = acts.shape[1]
    result = np.zeros((hidden, num_classes), dtype=np.float64)
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            result[:, cls] = acts[mask].mean(axis=0)
    return result
