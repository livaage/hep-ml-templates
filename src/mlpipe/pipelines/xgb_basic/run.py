from pathlib import Path
from mlpipe.core.config import load_pipeline_config
from mlpipe.core.registry import get
from mlpipe.core.utils import maybe_make_demo_csv

def run_pipeline(pipeline: str, config_path: str, config_name: str, overrides=None):
    assert pipeline == "xgb_basic", f"Only xgb_basic implemented in pass #1, got {pipeline}"
    cfg = load_pipeline_config(Path(config_path), pipeline_name=config_name, overrides=overrides or [])

    # 1) ingest
    data_cfg = cfg["data"]
    path = data_cfg["path"]
    if "demo_tabular.csv" in str(path):
        maybe_make_demo_csv(path)
    Ingest = get(data_cfg["block"])               # "ingest.csv"
    ing = Ingest(path=path, label=data_cfg["label"], **{k: v for k, v in data_cfg.items() if k not in ['block', 'path', 'label']})
    X, y = ing.load()

    # 2) feature engineering (optional)
    feat_cfg = cfg.get("feature_eng", {})
    if feat_cfg:
        Sel = get(feat_cfg["block"])              # "feature.column_selector"
        sel = Sel(include=feat_cfg.get("include"), exclude=feat_cfg.get("exclude"))
        X = sel.transform(X)

    # 3) preprocessing
    pre_cfg = cfg["preprocessing"]
    Pre = get(pre_cfg["block"])                   # "preprocessing.standard_scaler"
    prep = Pre().fit(X, y)
    Xp = prep.transform(X)

    # 4) model
    m_cfg = cfg["model"]
    Model = get(m_cfg["block"])                   # "model.xgb_classifier"
    model = Model()
    model.build(m_cfg.get("params", {}))

    # 5) train
    t_cfg = cfg["training"]
    Trainer = get(t_cfg["block"])                 # "train.sklearn"
    trainer = Trainer()
    model = trainer.train(model, Xp, y, t_cfg.get("params", {}))

    # 6) evaluate
    e_cfg = cfg["evaluation"]
    Eval = get(e_cfg["block"])                    # "eval.classification"
    evaluator = Eval()
    y_pred = model.predict(Xp)
    metrics = evaluator.evaluate(y, y_pred, e_cfg.get("params", {}))

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if v == v else f"{k}: NaN")
