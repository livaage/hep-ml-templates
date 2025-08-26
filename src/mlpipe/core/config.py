from pathlib import Path
from typing import Any, Dict, List
from omegaconf import OmegaConf


def load_yaml(path: Path) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore


def merge_overrides(cfg: Dict[str, Any], dotlist: List[str]) -> Dict[str, Any]:
    if not dotlist:
        return cfg
    base = OmegaConf.create(cfg)
    overrides = OmegaConf.from_dotlist(dotlist)
    merged = OmegaConf.merge(base, overrides)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore


def load_pipeline_config(config_path: Path, pipeline_name: str,
                         overrides: List[str] | None = None) -> Dict[str, Any]:
    """
    pipeline.yaml declares which group files to load:
      data: csv_demo
      preprocessing: standard
      feature_eng: column_selector
      model: xgb_classifier
      training: sklearn
      evaluation: classification

    Overrides can change which configs to load for each group:
      overrides=["data=higgs_uci", "model=different_model"]
    """
    cfg_root = config_path.resolve()
    pipe = load_yaml(cfg_root / f"{pipeline_name}.yaml")

    # Apply overrides to the pipeline config first (which configs to load)
    if overrides:
        pipe_overrides = {}
        final_overrides = []

        for override in overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                # If it's a top-level group (data, model, etc.), override which config to load
                if key in ["data", "preprocessing", "feature_eng", "model",
                           "training", "evaluation", "runtime"]:
                    pipe_overrides[key] = value
                else:
                    # Otherwise, it's a deep override for the final config
                    final_overrides.append(override)

        # Update the pipeline config with group overrides
        pipe.update(pipe_overrides)
        overrides = final_overrides  # Keep only deep overrides for later

    def grp(group: str, name: str):
        return load_yaml(cfg_root / group / f"{name}.yaml")

    final = {
        "data": grp("data", pipe["data"]),
        "preprocessing": grp("preprocessing", pipe["preprocessing"]),
        "feature_eng": grp("feature_eng", pipe["feature_eng"]),
        "model": grp("model", pipe["model"]),
        "training": grp("training", pipe["training"]),
        "evaluation": grp("evaluation", pipe["evaluation"]),
        "runtime": load_yaml(cfg_root / "runtime" / f"{pipe.get('runtime','local_cpu')}.yaml"),
    }
    # Apply any remaining deep overrides
    return merge_overrides(final, overrides or [])
