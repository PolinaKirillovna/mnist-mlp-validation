"""Tests for configuration loading and coercion."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnist_validation.config import Config, load_config


def test_default_config() -> None:
    config = Config()
    assert config.seed == 42
    assert config.model.hidden_sizes == (256, 128)
    assert config.paths.train_path == Path("data/raw/d3.csv")


def test_load_config_from_yaml(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    text = """
seed: 7
model:
  hidden_sizes: [32, 16]
  activation: gelu
  dropout_p: 0.5
grid:
  datasets: [balanced]
"""
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_config(path)
    assert config.seed == 7
    assert config.model.hidden_sizes == (32, 16)
    assert config.model.activation == "gelu"
    assert config.grid.datasets == ("balanced",)
    # Unspecified fields keep their defaults.
    assert config.training.epochs == 30


def test_load_config_none_returns_defaults() -> None:
    assert load_config(None).seed == 42


def test_load_config_rejects_unknown_key(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    path = tmp_path / "c.yaml"
    path.write_text("nonsense_key: 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Unknown config key"):
        load_config(path)
