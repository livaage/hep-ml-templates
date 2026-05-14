"""Core framework smoke tests: registry, config loading, basic block instantiation."""

from __future__ import annotations

from pathlib import Path

from mlpipe.core.config import load_pipeline_config
from mlpipe.core.registry import get, list_blocks


def test_registry_lists_expected_blocks():
    """The registry should be populated by importing mlpipe.blocks (via conftest)."""
    blocks = set(list_blocks())
    # Sanity-check a representative slice across categories; the full list grows
    # as optional deps come and go.
    for required in {
        "ingest.csv",
        "preprocessing.standard_scaler",
        "preprocessing.data_split",
        "feature.column_selector",
        "model.decision_tree",
        "model.random_forest",
        "model.svm",
        "model.mlp",
        "train.sklearn",
        "eval.classification",
    }:
        assert required in blocks, f"{required!r} missing from registry"


def test_get_returns_callable_class():
    cls = get("model.decision_tree")
    assert callable(cls)
    instance = cls()
    # ModelBlock instances expose build/fit/predict
    for method in ("build", "fit", "predict"):
        assert callable(getattr(instance, method)), f"{method} missing"


def test_pipeline_config_loads(configs_dir: Path):
    cfg = load_pipeline_config(configs_dir, pipeline_name="pipeline")
    assert isinstance(cfg, dict)
    for section in ("data", "preprocessing", "model", "training", "evaluation"):
        assert section in cfg, f"pipeline.yaml missing {section!r} section"


def test_pipeline_config_overrides(configs_dir: Path):
    """Top-level group overrides should pick a different config file."""
    cfg = load_pipeline_config(
        configs_dir,
        pipeline_name="pipeline",
        overrides=["model=decision_tree"],
    )
    assert cfg["model"]["block"] == "model.decision_tree"
