"""Framework-agnostic classifier interface.

Evaluation and interpretation depend only on this :class:`Classifier` protocol,
never on PyTorch or JAX directly. Implementations wrap a trained model and expose
logits and first-layer internals as plain NumPy arrays.

Convention: ``x`` is a float array of shape ``(N, 784)`` with pixel values scaled
to ``[0, 1]`` (the same normalisation used during training).
"""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Classifier(Protocol):
    """A trained classifier that exposes logits and first-layer internals."""

    def predict_logits(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class logits for a batch of normalised inputs.

        Args:
            x: Array of shape ``(N, 784)`` with values in ``[0, 1]``.

        Returns:
            Array of shape ``(N, num_classes)`` with raw logits.
        """
        ...

    def first_layer_weights(self) -> NDArray[np.float64]:
        """Return the first-layer weight matrix of shape ``(H, 784)``."""
        ...

    def first_layer_preactivations(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return first-layer pre-activations ``x W^T + b`` of shape ``(N, H)``.

        Args:
            x: Array of shape ``(N, 784)`` with values in ``[0, 1]``.

        Returns:
            Array of shape ``(N, H)``.
        """
        ...


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return row-wise softmax probabilities of ``logits``.

    Softmax is applied here, at inference, to obtain probabilities; during
    training the loss consumes raw logits (log-softmax is applied inside the loss).

    Args:
        logits: Array of shape ``(N, num_classes)``.

    Returns:
        Array of shape ``(N, num_classes)`` of probabilities that sum to 1 per row.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return cast("NDArray[np.float64]", exp / exp.sum(axis=1, keepdims=True))


def predict_labels(classifier: Classifier, x: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return argmax class predictions for ``x``.

    Args:
        classifier: Any :class:`Classifier`.
        x: Array of shape ``(N, 784)`` with values in ``[0, 1]``.

    Returns:
        Array of shape ``(N,)`` with predicted class indices.
    """
    return cast("NDArray[np.int64]", classifier.predict_logits(x).argmax(axis=1).astype(np.int64))
