"""JAX/Flax two-layer perceptron and its :class:`Classifier` adapter.

Mirrors :mod:`mnist_validation.models.torch_mlp` as closely as the frameworks
allow: same architecture, same activations, logits-only output (softmax applied
separately). Flax is functional — parameters live outside the module and are
threaded explicitly — and dropout consumes an explicit PRNG key, which are the
main structural differences from the imperative PyTorch version.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from mnist_validation.config import NUM_CLASSES
from mnist_validation.models.base import Classifier

_ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "gelu": nn.gelu,
}


class JaxMLP(nn.Module):
    """Flax two-hidden-layer perceptron producing logits.

    Attributes:
        hidden_sizes: Sizes of the hidden layers.
        activation: Activation name (``relu``, ``tanh`` or ``gelu``).
        dropout_p: Dropout probability after each hidden activation.
        num_classes: Number of output classes.
    """

    hidden_sizes: tuple[int, ...]
    activation: str
    dropout_p: float
    num_classes: int = NUM_CLASSES

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        """Return logits for a batch ``x`` of shape ``(N, 784)``.

        Args:
            x: Input batch.
            train: Whether dropout is active (training) or disabled (inference).

        Returns:
            Logits of shape ``(N, num_classes)``.
        """
        act = _ACTIVATIONS[self.activation]
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = act(x)
            if self.dropout_p > 0.0:
                x = nn.Dropout(rate=self.dropout_p, deterministic=not train)(x)
        return cast("jax.Array", nn.Dense(self.num_classes)(x))


def init_params(model: JaxMLP, seed: int, in_features: int) -> Any:
    """Initialise model parameters with a seeded PRNG key.

    Args:
        model: The Flax module.
        seed: Seed for the parameter-init PRNG key.
        in_features: Number of input features.

    Returns:
        The initialised parameter pytree (the ``params`` collection).
    """
    key = jax.random.PRNGKey(seed)
    variables = model.init(
        {"params": key, "dropout": key}, jnp.zeros((1, in_features)), train=False
    )
    return variables["params"]


class JaxClassifier(Classifier):
    """Adapter exposing trained Flax params through the classifier protocol.

    Attributes:
        model: The Flax module.
        params: The trained parameter pytree.
    """

    def __init__(self, model: JaxMLP, params: Any) -> None:
        """Initialise the adapter.

        Args:
            model: The Flax module.
            params: Trained parameters.
        """
        self.model = model
        self.params = params

    def predict_logits(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return logits for normalised inputs of shape ``(N, 784)``."""
        logits = self.model.apply(
            {"params": self.params}, jnp.asarray(x, dtype=jnp.float32), train=False
        )
        return np.asarray(logits, dtype=np.float64)

    def _first_dense(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self.params["Dense_0"])

    def first_layer_weights(self) -> NDArray[np.float64]:
        """Return the first-layer weight matrix of shape ``(H, 784)``."""
        kernel = np.asarray(self._first_dense()["kernel"], dtype=np.float64)  # (784, H)
        return kernel.T

    def first_layer_preactivations(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return first-layer pre-activations of shape ``(N, H)``."""
        dense = self._first_dense()
        kernel = np.asarray(dense["kernel"], dtype=np.float64)  # (784, H)
        bias = np.asarray(dense["bias"], dtype=np.float64)
        return x @ kernel + bias
