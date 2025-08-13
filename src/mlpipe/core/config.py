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

def load_pipeline_config(config_path: Path, pipeline_name: str, overrides: List[str] | None = None) -> Dict[str, Any]:
    """
    pipeline.yaml declares which group files to load:
      data: csv_demo
      preprocessing: standard
      feature_eng: column_selector
      model: xgb_classifier
      training: sklearn
      evaluation: classification
    """
    cfg_root = config_path.resolve()
    pipe = load_yaml(cfg_root / f"{pipeline_name}.yaml")

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
    # allow dotlist overrides
    return merge_overrides(final, overrides or [])
