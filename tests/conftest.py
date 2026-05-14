"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Trigger block registration once so `get(...)` works in every test.
import mlpipe.blocks  # noqa: F401


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    return repo_root / "data"


@pytest.fixture
def chdir_repo(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir into the repo root so configs with relative `data/...` paths resolve."""
    monkeypatch.chdir(repo_root)
    return repo_root


@pytest.fixture
def tiny_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """50 samples, 5 features, balanced binary target — enough for fit/predict smoke tests."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X.sum(axis=1) > 0).astype(int), name="label")
    return X, y


@pytest.fixture
def tiny_continuous_data() -> pd.DataFrame:
    """50 samples, 8 features — for autoencoder/reconstruction tests."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((50, 8)), columns=[f"f{i}" for i in range(8)])


# Allow pytest's plugin autoload (the previous `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` broke pytest-cov)
os.environ.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
