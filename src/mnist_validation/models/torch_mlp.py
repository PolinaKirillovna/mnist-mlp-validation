"""PyTorch two-layer perceptron and its :class:`Classifier` adapter.

The model outputs raw **logits**; softmax is not part of the model. During
training the loss (``nn.CrossEntropyLoss``) applies log-softmax internally, and
probabilities are obtained separately at inference via
:func:`~mnist_validation.models.base.softmax`.

"Two-layer perceptron" is interpreted as two *trainable hidden layers* followed
by the output layer (the assignment speaks of hidden layers in the plural); the
base configuration uses ``hidden_sizes=(256, 128)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from mnist_validation.config import NUM_CLASSES, NUM_PIXELS, ModelConfig
from mnist_validation.models.base import Classifier

logger = logging.getLogger(__name__)

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def make_activation(name: str) -> nn.Module:
    """Return a new activation module by name.

    Args:
        name: One of ``relu``, ``tanh`` or ``gelu``.

    Returns:
        A fresh activation module.

    Raises:
        KeyError: If the activation name is unknown.
    """
    if name not in _ACTIVATIONS:
        raise KeyError(f"Unknown activation '{name}', expected one of {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


def resolve_device(policy: str) -> torch.device:
    """Resolve a device policy to a concrete ``torch.device``.

    Args:
        policy: ``auto`` (mps -> cuda -> cpu) or an explicit device string.

    Returns:
        The resolved device.
    """
    if policy != "auto":
        return torch.device(policy)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    """Two-hidden-layer perceptron for flattened MNIST images.

    Attributes:
        net: The underlying sequential network producing logits.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 128),
        activation: str = "relu",
        dropout_p: float = 0.0,
        in_features: int = NUM_PIXELS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initialise the network.

        Args:
            hidden_sizes: Sizes of the hidden layers.
            activation: Activation function name.
            dropout_p: Dropout probability after each hidden activation.
            in_features: Number of input features (784 for MNIST).
            num_classes: Number of output classes.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(make_activation(activation))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of inputs of shape ``(N, in_features)``."""
        return cast("torch.Tensor", self.net(x))


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TorchClassifier(Classifier):
    """Adapter exposing a trained :class:`MLP` through the classifier protocol.

    Attributes:
        model: The wrapped PyTorch module (in eval mode for inference).
        device: Device used for inference.
        batch_size: Batch size for chunked inference.
    """

    def __init__(self, model: MLP, device: torch.device, batch_size: int = 1024) -> None:
        """Initialise the adapter.

        Args:
            model: A trained :class:`MLP`.
            device: Device to run inference on.
            batch_size: Batch size for chunked inference.
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

    def predict_logits(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return logits for normalised inputs of shape ``(N, 784)``."""
        self.model.eval()
        outputs: list[NDArray[np.float64]] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(x[start : start + self.batch_size], dtype=np.float32)
                ).to(self.device)
                outputs.append(self.model(chunk).cpu().numpy().astype(np.float64))
        return np.concatenate(outputs, axis=0)

    def _first_linear(self) -> nn.Linear:
        for module in self.model.net:
            if isinstance(module, nn.Linear):
                return module
        raise RuntimeError("No linear layer found in model")

    def first_layer_weights(self) -> NDArray[np.float64]:
        """Return the first-layer weight matrix of shape ``(H, 784)``."""
        weight = self._first_linear().weight.detach().cpu().numpy()
        return cast("NDArray[np.float64]", weight.astype(np.float64))

    def first_layer_preactivations(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return first-layer pre-activations of shape ``(N, H)``."""
        linear = self._first_linear()
        weight = linear.weight.detach().cpu().numpy().astype(np.float64)
        bias = linear.bias.detach().cpu().numpy().astype(np.float64)
        return cast("NDArray[np.float64]", x @ weight.T + bias)


def save_torch_model(
    classifier: TorchClassifier, model_config: ModelConfig, path: str | Path
) -> Path:
    """Persist a trained model with the architecture needed to rebuild it.

    Args:
        classifier: Trained classifier to save.
        model_config: Architecture configuration.
        path: Output ``.pt`` path.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": classifier.model.state_dict(),
            "hidden_sizes": list(model_config.hidden_sizes),
            "activation": model_config.activation,
            "dropout_p": model_config.dropout_p,
        },
        out,
    )
    return out


def load_torch_model(path: str | Path, device: torch.device) -> TorchClassifier:
    """Load a model saved with :func:`save_torch_model`.

    Args:
        path: Path to the ``.pt`` checkpoint.
        device: Device to load the model onto.

    Returns:
        A :class:`TorchClassifier` ready for inference.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    model = MLP(
        hidden_sizes=tuple(payload["hidden_sizes"]),
        activation=payload["activation"],
        dropout_p=payload["dropout_p"],
    )
    model.load_state_dict(payload["state_dict"])
    return TorchClassifier(model, device)
