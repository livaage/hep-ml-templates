"""End-to-end pipeline tests.

Each test invokes `run_pipeline` in-process against the bundled configs and
demo data, then asserts on the returned `metrics` dict. No subprocess, no
`pip install`, no copying the source tree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mlpipe.core.universal_runner import run_pipeline


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _run(configs_dir: Path, overrides: list[str]) -> dict:
    """Call run_pipeline with sensible defaults; return the result dict."""
    return run_pipeline(
        pipeline="auto",
        config_path=str(configs_dir),
        config_name="pipeline",
        overrides=overrides,
    )


@pytest.mark.parametrize(
    "model_override",
    [
        # XGBoost is in the default pipeline.yaml, but pin it explicitly anyway
        pytest.param(
            "xgb_classifier",
            marks=pytest.mark.skipif(not _has("xgboost"), reason="xgboost not installed"),
            id="xgb_classifier",
        ),
        pytest.param("decision_tree", id="decision_tree"),
        pytest.param("random_forest", id="random_forest"),
        pytest.param("mlp", id="mlp"),
        pytest.param("ensemble_voting", id="ensemble_voting"),
    ],
)
def test_sklearn_classification_pipeline(chdir_repo, configs_dir, model_override):
    """Each classification pipeline should run end-to-end and return metrics."""
    result = _run(configs_dir, overrides=[f"model={model_override}"])
    assert "metrics" in result and result["metrics"], "pipeline must return non-empty metrics"
    assert "model" in result
    # The default evaluator is classification; accuracy should be a finite float.
    metrics = result["metrics"]
    if "accuracy" in metrics:
        assert 0.0 <= float(metrics["accuracy"]) <= 1.0


_TORCH_AVAILABLE = _has("torch") and _has("pytorch_lightning")


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch + lightning not installed")
@pytest.mark.parametrize("model_override", ["ae_vanilla", "ae_variational"])
def test_autoencoder_pipeline(chdir_repo, configs_dir, model_override):
    """Autoencoder pipelines should fit + reconstruct + return loss metrics."""
    result = _run(
        configs_dir,
        overrides=[
            f"model={model_override}",
            "training=pytorch",
            "evaluation=reconstruction",
            # Keep the test fast — full pipeline.yaml defaults train for many epochs.
            "model.params.max_epochs=2",
            "model.params.batch_size=16",
            "model.params.encoder_layers=[8]",
            "model.params.decoder_layers=[8]",
            "model.params.latent_dim=4",
        ],
    )
    assert "metrics" in result and result["metrics"], "AE pipeline must return metrics"


@pytest.mark.skipif(not _has("torch_geometric"), reason="torch_geometric not installed")
def test_gnn_pipeline_runs_or_surfaces_clear_error(chdir_repo, configs_dir):
    """If torch_geometric is importable, the GNN pipeline should run end-to-end."""
    result = _run(
        configs_dir,
        overrides=[
            "data=graph_demo",
            "model=gnn_gcn",
            "model.params.task=node",
        ],
    )
    assert "metrics" in result


def test_gnn_pipeline_raises_helpful_error_without_torch_geometric(
    chdir_repo, configs_dir, monkeypatch
):
    """Without torch_geometric, the runner should raise a RuntimeError mentioning it."""
    if _has("torch_geometric"):
        pytest.skip("torch_geometric is installed; this test only covers the missing-dep path")

    with pytest.raises(RuntimeError, match=r"[Tt]orch[ _]?[Gg]eometric"):
        _run(
            configs_dir,
            overrides=[
                "data=graph_demo",
                "model=gnn_gcn",
                "model.params.task=node",
            ],
        )
