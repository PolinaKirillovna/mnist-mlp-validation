"""Gradient-based sensitivity methods (additional Part 4 analysis).

Implements Saliency (``|d logit / d x|``), Input x Gradient and Integrated
Gradients, using autograd in **both** frameworks. Each framework provides an
"input gradient" function ``(image, target) -> d logit_target / d x``; the three
methods are then framework-agnostic combinators over that gradient.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from numpy.typing import NDArray

from mnist_validation.models.jax_mlp import JaxClassifier
from mnist_validation.models.torch_mlp import TorchClassifier

# (image[784], target) -> gradient[784] of the target logit w.r.t. the input.
InputGradientFn = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


def torch_input_gradient(classifier: TorchClassifier) -> InputGradientFn:
    """Return an input-gradient function backed by PyTorch autograd."""

    def grad_fn(image: NDArray[np.float64], target: int) -> NDArray[np.float64]:
        classifier.model.eval()
        x = (
            torch.tensor(image, dtype=torch.float32, device=classifier.device)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        logits = classifier.model(x)
        logits[0, target].backward()
        assert x.grad is not None
        return cast("NDArray[np.float64]", x.grad[0].detach().cpu().numpy().astype(np.float64))

    return grad_fn


def jax_input_gradient(classifier: JaxClassifier) -> InputGradientFn:
    """Return an input-gradient function backed by ``jax.grad``."""

    def scalar(x: jax.Array, target: int) -> jax.Array:
        logits = cast(
            "jax.Array",
            classifier.model.apply({"params": classifier.params}, x[None, :], train=False),
        )
        return logits[0, target]

    def grad_fn(image: NDArray[np.float64], target: int) -> NDArray[np.float64]:
        g = jax.grad(scalar)(jnp.asarray(image, dtype=jnp.float32), target)
        return np.asarray(g, dtype=np.float64)

    return grad_fn


def saliency(grad: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Saliency map ``|grad|``."""
    return np.abs(grad)


def input_x_gradient(image: NDArray[np.float64], grad: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Input x Gradient attribution ``image * grad``."""
    return image * grad


def integrated_gradients(
    grad_fn: InputGradientFn,
    image: NDArray[np.float64],
    target: int,
    steps: int = 32,
    baseline: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return the Integrated Gradients attribution for one image.

    Args:
        grad_fn: Input-gradient function for a framework.
        image: Normalised image of shape ``(784,)``.
        target: Target class index.
        steps: Number of Riemann steps along the path.
        baseline: Path baseline (defaults to zeros = black image).

    Returns:
        Attribution map of shape ``(784,)``.
    """
    if baseline is None:
        baseline = np.zeros_like(image)
    diff = image - baseline
    total = np.zeros_like(image)
    for step in range(1, steps + 1):
        alpha = step / steps
        total += grad_fn(baseline + alpha * diff, target)
    return diff * (total / steps)


def all_methods(
    grad_fn: InputGradientFn, image: NDArray[np.float64], target: int, ig_steps: int = 32
) -> dict[str, NDArray[np.float64]]:
    """Compute all three gradient attributions for one image.

    Args:
        grad_fn: Input-gradient function for a framework.
        image: Normalised image of shape ``(784,)``.
        target: Target class index.
        ig_steps: Integrated-Gradients Riemann steps.

    Returns:
        Mapping ``{"saliency", "input_x_gradient", "integrated_gradients"}`` to maps.
    """
    grad = grad_fn(image, target)
    return {
        "saliency": saliency(grad),
        "input_x_gradient": input_x_gradient(image, grad),
        "integrated_gradients": integrated_gradients(grad_fn, image, target, ig_steps),
    }
