"""Per-block unit tests.

Each block is built, then driven through its interface with a tiny synthetic
dataset. These are smoke tests — they catch "the block can't even fit/predict
without crashing" regressions in <1s each.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from mlpipe.core.registry import get


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


# --- Models ---------------------------------------------------------------

# Classifier blocks that should work with only core + sklearn (+ optional xgb)
SKLEARN_CLASSIFIERS = [
    "model.decision_tree",
    "model.random_forest",
    "model.svm",
    "model.mlp",
    "model.adaboost",
    "model.ensemble_voting",
]


@pytest.mark.parametrize("block_name", SKLEARN_CLASSIFIERS)
def test_sklearn_classifier_fit_predict(block_name, tiny_classification_data):
    X, y = tiny_classification_data
    Model = get(block_name)
    model = Model()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


@pytest.mark.skipif(not _has("xgboost"), reason="xgboost not installed")
def test_xgb_classifier_fit_predict(tiny_classification_data):
    X, y = tiny_classification_data
    Model = get("model.xgb_classifier")
    model = Model()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


# --- Preprocessing --------------------------------------------------------


def test_standard_scaler_fit_transform(tiny_classification_data):
    X, _ = tiny_classification_data
    Scaler = get("preprocessing.standard_scaler")
    out = Scaler().fit(X).transform(X)
    out = np.asarray(out)
    assert out.shape == X.shape
    # Means should be ~0 after scaling
    assert np.allclose(out.mean(axis=0), 0, atol=1e-6)


def test_data_split_produces_disjoint_splits(tiny_classification_data):
    X, y = tiny_classification_data
    Split = get("preprocessing.data_split")
    splitter = Split()
    splits = splitter.split(X, y)
    assert isinstance(splits, dict)
    # Whether it's two-way or three-way, no row should appear in more than one split.
    seen_indices: set[int] = set()
    total = 0
    for split_name, (X_part, _y_part) in splits.items():
        idx = set(X_part.index.tolist())
        assert idx.isdisjoint(seen_indices), f"{split_name!r} overlaps a previous split"
        seen_indices |= idx
        total += len(idx)
    assert total == len(X)


# --- Feature engineering --------------------------------------------------


def test_column_selector_include_subset(tiny_classification_data):
    X, _ = tiny_classification_data
    Selector = get("feature.column_selector")
    sel = Selector(include=["f0", "f1", "f2"])
    out = sel.transform(X)
    assert list(out.columns) == ["f0", "f1", "f2"]


# --- Ingest ---------------------------------------------------------------


def test_csv_loader_loads_bundled_demo(repo_root, data_dir):
    Loader = get("ingest.csv")
    loader = Loader(
        config={
            "file_path": str(data_dir / "demo_tabular.csv"),
            "target_column": "label",
        }
    )
    X, y, metadata = loader.load()
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(y) > 0
    assert isinstance(metadata, dict)


# --- Evaluation -----------------------------------------------------------


def test_classification_evaluator_returns_metrics(tiny_classification_data):
    X, y = tiny_classification_data
    # Predict on the perfect identity oracle to get a deterministic, sensible score.
    y_pred = y.values.astype(float)
    Eval = get("eval.classification")
    metrics = Eval().evaluate(y, y_pred, {})
    assert isinstance(metrics, dict)
    assert metrics, "evaluator should return at least one metric"
    # Most classification metrics will be at their best when y_pred == y_true.
    if "accuracy" in metrics:
        assert metrics["accuracy"] >= 0.99


# --- Optional: torch/lightning autoencoders -------------------------------

_TORCH_AVAILABLE = _has("torch") and _has("pytorch_lightning")


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch + lightning not installed")
@pytest.mark.parametrize("block_name", ["model.ae_vanilla", "model.ae_variational"])
def test_autoencoder_builds_and_fits(block_name, tiny_continuous_data):
    Model = get(block_name)
    # Force a tiny, fast architecture and 1 epoch — this is a smoke test only.
    model = Model(
        encoder_layers=[8],
        latent_dim=4,
        decoder_layers=[8],
        max_epochs=1,
        batch_size=16,
        normalize_inputs=True,
    )
    model.fit(tiny_continuous_data)
    # AE should expose a reconstruct or predict method
    if hasattr(model, "reconstruct"):
        out = model.reconstruct(tiny_continuous_data)
    else:
        out = model.predict(tiny_continuous_data)
    out = np.asarray(out)
    assert out.shape[0] == len(tiny_continuous_data)
