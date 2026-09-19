"""Occlusion sensitivity maps.

The image is repeatedly modified by replacing a ``k x k`` window (stride 1) with
a fixed background value; after each modification the model re-predicts. For each
window position we record how much the probability of the predicted (and true)
class drops relative to the unoccluded image — a large drop means the occluded
region was important.

Mapping windows to pixels: each window's scalar drop is **accumulated into a
28x28 buffer** over every pixel it covers, together with a coverage counter, and
the final map is the per-pixel average (buffer / counter). All occluded variants
are evaluated in a single batched forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mnist_validation.config import IMAGE_SIDE
from mnist_validation.models.base import Classifier, softmax


@dataclass(frozen=True)
class OcclusionResult:
    """Occlusion maps for a single image and kernel size.

    Attributes:
        kernel: Occlusion window size.
        predicted_class: Model's predicted class on the unoccluded image.
        true_class: Ground-truth class.
        base_prob_pred: Base probability of the predicted class.
        base_prob_true: Base probability of the true class.
        importance_pred: 28x28 map of probability drop for the predicted class.
        importance_true: 28x28 map of probability drop for the true class.
    """

    kernel: int
    predicted_class: int
    true_class: int
    base_prob_pred: float
    base_prob_true: float
    importance_pred: NDArray[np.float64]
    importance_true: NDArray[np.float64]


def _window_positions(side: int, kernel: int, stride: int) -> list[tuple[int, int]]:
    """Return the (row, col) top-left positions of all occlusion windows."""
    rows = range(0, side - kernel + 1, stride)
    cols = range(0, side - kernel + 1, stride)
    return [(r, c) for r in rows for c in cols]


def _build_occluded_batch(
    image_2d: NDArray[np.float64],
    positions: list[tuple[int, int]],
    kernel: int,
    fill: float,
) -> NDArray[np.float64]:
    """Return a ``(P, 784)`` batch with each window occluded by ``fill``."""
    batch = np.repeat(image_2d.reshape(1, -1), len(positions), axis=0)
    for i, (r, c) in enumerate(positions):
        patch = batch[i].reshape(IMAGE_SIDE, IMAGE_SIDE)
        patch[r : r + kernel, c : c + kernel] = fill
    return batch


def _accumulate_map(
    drops: NDArray[np.float64], positions: list[tuple[int, int]], kernel: int
) -> NDArray[np.float64]:
    """Average per-window drops into a 28x28 per-pixel map via a coverage buffer."""
    buffer = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float64)
    counter = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float64)
    for drop, (r, c) in zip(drops, positions, strict=True):
        buffer[r : r + kernel, c : c + kernel] += drop
        counter[r : r + kernel, c : c + kernel] += 1.0
    counter[counter == 0.0] = 1.0
    return buffer / counter


def occlusion_map(
    classifier: Classifier,
    image: NDArray[np.float64],
    true_label: int,
    kernel: int,
    stride: int = 1,
    fill: float = 0.0,
) -> OcclusionResult:
    """Compute occlusion sensitivity maps for one image.

    Args:
        classifier: Any :class:`Classifier`.
        image: Normalised image of shape ``(784,)`` with values in ``[0, 1]``.
        true_label: Ground-truth class of the image.
        kernel: Occlusion window size (e.g. 1, 2 or 3).
        stride: Sliding stride.
        fill: Fill value for the occluded window (0.0 = background).

    Returns:
        An :class:`OcclusionResult`.
    """
    image = image.astype(np.float64)
    base_probs = softmax(classifier.predict_logits(image.reshape(1, -1)))[0]
    predicted = int(base_probs.argmax())

    positions = _window_positions(IMAGE_SIDE, kernel, stride)
    batch = _build_occluded_batch(image.reshape(IMAGE_SIDE, IMAGE_SIDE), positions, kernel, fill)
    occ_probs = softmax(classifier.predict_logits(batch))

    drop_pred = base_probs[predicted] - occ_probs[:, predicted]
    drop_true = base_probs[true_label] - occ_probs[:, true_label]

    return OcclusionResult(
        kernel=kernel,
        predicted_class=predicted,
        true_class=int(true_label),
        base_prob_pred=float(base_probs[predicted]),
        base_prob_true=float(base_probs[true_label]),
        importance_pred=_accumulate_map(drop_pred, positions, kernel),
        importance_true=_accumulate_map(drop_true, positions, kernel),
    )
