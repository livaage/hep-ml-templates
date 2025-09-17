"""Pipeline configuration generator for hep-ml-templates.
Creates pipeline.yaml files dynamically based on user choices.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Pipeline configurations for different algorithms
PIPELINE_CONFIGS = {
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
    "torch": {
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

# Default components that can be used across pipelines
DEFAULT_COMPONENTS = {
    "data": "csv_demo",
    "preprocessing": "standard",
    "feature_eng": "all_columns",
    "training": "sklearn",
    "evaluation": "classification",
    "runtime": "local_cpu",
}


def generate_pipeline_config(
    pipeline_type: str,
    custom_components: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate a pipeline configuration for a specific algorithm type.

    Args:
        pipeline_type: Type of pipeline (decision-tree, xgb, neural, torch, gnn)
        custom_components: Optional dict to override default components
        output_path: Optional path to write the config to

    Returns:
        Dictionary containing the pipeline configuration
    """
    # Start with the pipeline template
    if pipeline_type in PIPELINE_CONFIGS:
        config = PIPELINE_CONFIGS[pipeline_type].copy()
    else:
        # For unknown pipeline types, use defaults with user-specified model
        config = DEFAULT_COMPONENTS.copy()
        config["model"] = pipeline_type  # Assume pipeline_type is the model name

    # Apply any custom component overrides
    if custom_components:
        config.update(custom_components)

    # Write to file if path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✅ Generated pipeline config: {output_path}")

    return config


def get_pipeline_dependencies(pipeline_type: str) -> Dict[str, list]:
    """Get the dependencies required for a specific pipeline type.

    Args:
        pipeline_type: Type of pipeline (decision-tree, xgb, neural, torch, gnn)

    Returns:
        Dict with 'required' and 'optional' dependency lists
    """
    base_deps = ["omegaconf>=2.3", "numpy>=1.22", "pandas>=2.0", "scikit-learn>=1.2"]

    pipeline_specific_deps = {
        "decision-tree": {"required": []},
        "xgb": {"required": ["xgboost>=1.7"]},
        "neural": {"required": []},
        "torch": {"required": ["torch>=2.2", "lightning>=2.2"]},
        "gnn": {"required": ["torch>=2.2", "torch-geometric>=2.5"]},
    }

    # Optional dependencies based on data ingestion method
    optional_deps = {
        "uproot": ["uproot>=5.0", "awkward>=2.0"],  # For ROOT file ingestion
        "requests": ["requests>=2.25"],  # For downloading datasets
    }

    required = base_deps + pipeline_specific_deps.get(pipeline_type, {}).get("required", [])

    return {"required": required, "optional": optional_deps}


def detect_required_dependencies(config: Dict[str, Any]) -> list:
    """Analyze a pipeline configuration to detect required dependencies.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        List of required dependency strings
    """
    deps = ["omegaconf>=2.3", "numpy>=1.22", "pandas>=2.0", "scikit-learn>=1.2"]

    # Check model dependencies
    model = config.get("model", "")
    if "xgb" in model:
        deps.append("xgboost>=1.7")
    elif any(neural in model for neural in ["torch", "lightning", "ae_"]):
        deps.extend(["torch>=2.2", "lightning>=2.2"])
    elif "gnn" in model:
        deps.extend(["torch>=2.2", "torch-geometric>=2.5"])

    # Check data ingestion dependencies
    data = config.get("data", "")
    # This would need to be extended based on actual data config inspection
    # For now, we'll handle this in the data configuration files themselves

    return list(set(deps))  # Remove duplicates


def list_available_pipelines() -> Dict[str, Dict[str, Any]]:
    """List all available pipeline configurations with their descriptions.

    Returns:
        Dict mapping pipeline names to their configs and metadata
    """
    pipelines = {}

    for name, config in PIPELINE_CONFIGS.items():
        deps = get_pipeline_dependencies(name)
        pipelines[name] = {
            "config": config,
            "dependencies": deps,
            "description": f"{name.title()} pipeline with {config['model']} model",
        }

    return pipelines


if __name__ == "__main__":
    # Example usage
    print("Available pipelines:")
    for name, info in list_available_pipelines().items():
        print(f"  {name}: {info['description']}")
        print(f"    Dependencies: {info['dependencies']['required']}")
        print()
