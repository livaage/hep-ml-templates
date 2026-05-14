"""Pipeline configuration generator: writes pipeline.yaml for a chosen template."""

from pathlib import Path
from typing import Any

import yaml

# Canonical slot mapping for each pipeline template. Used both here (to write
# pipeline.yaml) and by mlpipe.cli.local_install (to decide which blocks and
# configs to copy when scaffolding a project).
PIPELINE_CONFIGS: dict[str, dict[str, str]] = {
    "decision-tree": {
        "data": "csv_demo",
        "preprocessing": "standard",
        "feature_eng": "all_columns",
        "model": "decision_tree",
        "training": "sklearn",
        "evaluation": "classification",
        "runtime": "local_cpu",
    },
    "xgb": {
        "data": "csv_demo",
        "preprocessing": "standard",
        "feature_eng": "all_columns",
        "model": "xgb_classifier",
        "training": "sklearn",
        "evaluation": "classification",
        "runtime": "local_cpu",
    },
    "ensemble": {
        "data": "csv_demo",
        "preprocessing": "standard",
        "feature_eng": "all_columns",
        "model": "ensemble_voting",
        "training": "sklearn",
        "evaluation": "classification",
        "runtime": "local_cpu",
    },
    "neural": {
        "data": "csv_demo",
        "preprocessing": "standard",
        "feature_eng": "all_columns",
        "model": "mlp",
        "training": "sklearn",
        "evaluation": "classification",
        "runtime": "local_cpu",
    },
    "autoencoder": {
        "data": "csv_demo",
        "preprocessing": "standard",
        "feature_eng": "all_columns",
        "model": "ae_vanilla",
        "training": "pytorch",
        "evaluation": "reconstruction",
        "runtime": "local_cpu",
    },
    "autoencoder-lightning": {
        "data": "csv_demo",
        "preprocessing": "standard",
        "feature_eng": "all_columns",
        "model": "ae_lightning",
        "training": "pytorch",
        "evaluation": "reconstruction",
        "runtime": "local_cpu",
    },
    "gnn": {
        "data": "graph_demo",
        "preprocessing": "standard",
        "feature_eng": "all_columns",
        "model": "gnn_pyg",
        "training": "sklearn",
        "evaluation": "classification",
        "runtime": "local_cpu",
    },
}

# Back-compat alias: older CLI usage referred to the Lightning autoencoder as "torch".
PIPELINE_CONFIGS["torch"] = PIPELINE_CONFIGS["autoencoder-lightning"]

# Default components used when a user passes a model name we don't have a
# template for (we fall back to a sklearn classification pipeline).
DEFAULT_COMPONENTS: dict[str, str] = {
    "data": "csv_demo",
    "preprocessing": "standard",
    "feature_eng": "all_columns",
    "training": "sklearn",
    "evaluation": "classification",
    "runtime": "local_cpu",
}


def generate_pipeline_config(
    pipeline_type: str,
    custom_components: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a pipeline configuration for a known template (or fall back to defaults).

    Args:
        pipeline_type: Template name (e.g. "xgb") or model name to substitute into
            the default sklearn classification template.
        custom_components: Optional slot overrides applied on top of the template.
        output_path: If given, the resulting config is also written to this path.

    Returns:
        The resolved pipeline configuration (a slot → config-name dict).
    """
    if pipeline_type in PIPELINE_CONFIGS:
        config = PIPELINE_CONFIGS[pipeline_type].copy()
    else:
        config = DEFAULT_COMPONENTS.copy()
        config["model"] = pipeline_type

    if custom_components:
        config.update(custom_components)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    return config


def list_available_pipelines() -> dict[str, dict[str, Any]]:
    """Return each available pipeline template with its slot configuration."""
    return {
        name: {
            "config": config,
            "description": f"{name.title()} pipeline with {config['model']} model",
        }
        for name, config in PIPELINE_CONFIGS.items()
    }
