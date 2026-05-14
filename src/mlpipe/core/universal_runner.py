"""Universal pipeline runner for hep-ml-templates.
Dynamically executes pipelines based on configuration without hardcoded implementations.
"""

from pathlib import Path

from mlpipe.core.config import load_pipeline_config
from mlpipe.core.registry import get
from mlpipe.core.utils import maybe_make_demo_csv


def run_pipeline(pipeline: str, config_path: str, config_name: str, overrides=None):
    """Run any pipeline configuration dynamically.

    Args:
        pipeline: Pipeline identifier (for future use/backwards compatibility)
        config_path: Path to configuration directory
        config_name: Name of pipeline config file (without .yaml)
        overrides: List of override strings for configuration
    """
    cfg = load_pipeline_config(
        Path(config_path), pipeline_name=config_name, overrides=overrides or []
    )

    print(f"Running pipeline: {config_name}")

    # 1) Data ingestion
    data_cfg = cfg["data"]
    path = data_cfg.get("path") or data_cfg.get("file_path")
    if path and "demo_tabular.csv" in str(path):
        maybe_make_demo_csv(path)

    Ingest = get(data_cfg["block"])
    X, y, metadata = Ingest(config=data_cfg).load()
    print(f"  Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 2) Feature engineering (optional)
    feat_cfg = cfg.get("feature_eng", {})
    if feat_cfg and feat_cfg.get("block"):
        Sel = get(feat_cfg["block"])
        X = Sel(include=feat_cfg.get("include"), exclude=feat_cfg.get("exclude")).transform(X)
        print(f"  Features selected: {X.shape[1]} columns")

    # 3) Preprocessing
    pre_cfg = cfg["preprocessing"]
    prep = get(pre_cfg["block"])().fit(X, y)
    Xp = prep.transform(X)

    # 4) Model
    m_cfg = cfg["model"]
    if m_cfg["block"].startswith("model.gnn_"):
        try:
            import torch_geometric  # noqa: F401
        except ImportError as err:
            raise RuntimeError(
                "Torch Geometric (torch_geometric) is required for GNN models. "
                "Install the extra dependencies, e.g.:\n\n"
                "  pip install -e '.[model-gnn]'\n\n"
                "Or follow the official wheels instructions: "
                "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            ) from err
    model = get(m_cfg["block"])()
    model.build(m_cfg.get("params", {}))

    # 5) Training
    t_cfg = cfg["training"]
    model = get(t_cfg["block"])().train(model, Xp, y, t_cfg.get("params", {}))

    # 6) Evaluation
    e_cfg = cfg["evaluation"]
    evaluator = get(e_cfg["block"])()
    if "reconstruction" in e_cfg["block"]:
        # Autoencoders evaluate on reconstructed input rather than labels
        y_pred = model.reconstruct(Xp) if hasattr(model, "reconstruct") else model.predict(Xp)
        metrics = evaluator.evaluate(Xp, y_pred, e_cfg.get("params", {}))
    else:
        y_pred = model.predict(Xp)
        metrics = evaluator.evaluate(y, y_pred, e_cfg.get("params", {}))

    print("\nResults:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nPipeline execution completed successfully")

    return {"model": model, "metrics": metrics, "preprocessor": prep, "metadata": metadata}


def validate_pipeline_config(config_path: Path, config_name: str) -> bool:
    """Validate that a pipeline configuration has all required components.

    Args:
        config_path: Path to configuration directory
        config_name: Name of pipeline config file

    Returns:
        True if valid, False otherwise
    """
    try:
        cfg = load_pipeline_config(config_path, config_name)

        required_sections = ["data", "preprocessing", "model", "training", "evaluation"]
        missing_sections = []

        for section in required_sections:
            if section not in cfg:
                missing_sections.append(section)
            elif not cfg[section].get("block"):
                missing_sections.append(f"{section}.block")

        if missing_sections:
            return False

        return True

    except Exception:
        return False


def get_pipeline_info(config_path: Path, config_name: str) -> dict:
    """Get information about a pipeline configuration.

    Args:
        config_path: Path to configuration directory
        config_name: Name of pipeline config file

    Returns:
        Dictionary with pipeline information
    """
    try:
        cfg = load_pipeline_config(config_path, config_name)

        info = {
            "data_source": cfg.get("data", {}).get("block", "unknown"),
            "model": cfg.get("model", {}).get("block", "unknown"),
            "preprocessing": cfg.get("preprocessing", {}).get("block", "unknown"),
            "feature_engineering": cfg.get("feature_eng", {}).get("block", "none"),
            "training": cfg.get("training", {}).get("block", "unknown"),
            "evaluation": cfg.get("evaluation", {}).get("block", "unknown"),
            "runtime": cfg.get("runtime", {}).get("block", "unknown"),
        }

        return info

    except Exception as e:
        return {"error": str(e)}
