"""Command-line entry points.

Usage: ``python -m mnist_validation <command> [--config path]``.

The CLI is the outermost layer: it wires configuration to the underlying
pipeline stages and owns user-facing output. Individual pipeline stages are
implemented in their respective layers and filled in stage by stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from mnist_validation.config import Config, load_config
from mnist_validation.data.loading import load_bundle

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for CLI runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_check_data(config: Config) -> int:
    """Load both datasets, validate their format and print a summary.

    Args:
        config: Loaded experiment configuration.

    Returns:
        Process exit code.
    """
    import numpy as np

    for path, name in ((config.paths.train_path, "train"), (config.paths.test_path, "test")):
        bundle = load_bundle(path, name)
        counts = np.bincount(bundle.labels, minlength=10)
        print(f"[{name}] {bundle.num_samples} samples, {bundle.images.shape[1]} features")
        print(
            f"[{name}] dtype={bundle.images.dtype}, "
            f"range=[{bundle.images.min()}, {bundle.images.max()}]"
        )
        print(f"[{name}] per-class counts: {counts.tolist()}")
    return 0


def _not_implemented(stage: str) -> int:
    """Report that a command is implemented in a later stage."""
    logger.warning("This command is implemented in %s and is not available yet.", stage)
    return 0


def cmd_validate(config: Config) -> int:
    """Part 1: data validation, anomalies, figures and tables."""
    from mnist_validation.pipeline import run_validation

    run_validation(config)
    return 0


def cmd_grid(config: Config) -> int:
    """Part 2: full PyTorch training grid."""
    return _not_implemented("stage 2 (training grid)")


def cmd_interpret(config: Config) -> int:
    """Part 3: error analysis, occlusion and first-layer maps."""
    return _not_implemented("stage 3 (interpretation)")


def cmd_jax_compare(config: Config) -> int:
    """Part 4: JAX runs and PyTorch/JAX comparison."""
    return _not_implemented("stage 4 (JAX comparison)")


def cmd_smoke(config: Config) -> int:
    """Fast end-to-end sanity run on a subsample."""
    return _not_implemented("stage 2 (training grid)")


_COMMANDS = {
    "check-data": cmd_check_data,
    "validate": cmd_validate,
    "grid": cmd_grid,
    "interpret": cmd_interpret,
    "jax-compare": cmd_jax_compare,
    "smoke": cmd_smoke,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="mnist-validation",
        description="MNIST MLP validation pipeline (PyTorch + JAX).",
    )
    parser.add_argument("command", choices=sorted(_COMMANDS), help="Pipeline command to run.")
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    config = load_config(args.config)
    return _COMMANDS[args.command](config)


if __name__ == "__main__":
    sys.exit(main())
