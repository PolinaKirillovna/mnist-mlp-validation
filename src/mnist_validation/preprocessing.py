"""Input preprocessing shared by both frameworks.

The only preprocessing step is scaling ``uint8`` pixels to ``float32`` in
``[0, 1]``. Keeping it in one place guarantees PyTorch and JAX see identical
inputs, which is required for a fair framework comparison.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mnist_validation.config import PIXEL_MAX


def to_float01(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Scale ``uint8`` images to ``float32`` in ``[0, 1]``.

    Args:
        images: Array of shape ``(N, 784)`` with values in ``[0, 255]``.

    Returns:
        ``float32`` array of the same shape with values in ``[0, 1]``.
    """
    return (images.astype(np.float32) / PIXEL_MAX).astype(np.float32)
