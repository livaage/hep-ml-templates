"""
Universal pipeline runner for hep-ml-templates.
Dynamically executes pipelines based on configuration without hardcoded implementations.
"""

from pathlib import Path
from mlpipe.core.config import load_pipeline_config
from mlpipe.core.registry import get
from mlpipe.core.utils import maybe_make_demo_csv


def run_pipeline(pipeline: str, config_path: str, config_name: str, overrides=None):
    """
    Run any pipeline configuration dynamically.
    
    Args:
        pipeline: Pipeline identifier (for future use/backwards compatibility)
        config_path: Path to configuration directory
        config_name: Name of pipeline config file (without .yaml)
        overrides: List of override strings for configuration
    """
    cfg = load_pipeline_config(
        Path(config_path), pipeline_name=config_name, overrides=overrides or [])

    print(f"🚀 Running pipeline with configuration: {config_name}")
    print(f"📁 Config path: {config_path}")
    if overrides:
        print(f"⚙️  Overrides: {overrides}")
    print()

    # 1) Data Ingestion
    print("📊 Loading data...")
    data_cfg = cfg["data"]
    
    # Handle both old and new config formats for backward compatibility
    path = data_cfg.get("path") or data_cfg.get("file_path")
    label = data_cfg.get("label") or data_cfg.get("target_column")

    # Create demo data if needed
    if path and "demo_tabular.csv" in str(path):
        maybe_make_demo_csv(path)

    try:
        Ingest = get(data_cfg["block"])
        ing = Ingest(config=data_cfg)
        X, y, metadata = ing.load()
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

    # 2) Feature Engineering (optional)
    feat_cfg = cfg.get("feature_eng", {})
    if feat_cfg and feat_cfg.get("block"):
        print("🔧 Applying feature engineering...")
        try:
            Sel = get(feat_cfg["block"])
            sel = Sel(include=feat_cfg.get("include"), exclude=feat_cfg.get("exclude"))
            X = sel.transform(X)
            print(f"✅ Features selected: {X.shape[1]} features remaining")
        except Exception as e:
            print(f"❌ Error in feature engineering: {e}")
            raise

    # 3) Preprocessing
    print("🔄 Preprocessing data...")
    try:
        pre_cfg = cfg["preprocessing"]
        Pre = get(pre_cfg["block"])
        prep = Pre().fit(X, y)
        Xp = prep.transform(X)
        print("✅ Data preprocessing completed")
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        raise

    # 4) Model Building
    print("🤖 Building model...")
    try:
        m_cfg = cfg["model"]
        Model = get(m_cfg["block"])
        model = Model()
        model.build(m_cfg.get("params", {}))
        print(f"✅ Model built: {m_cfg['block']}")
    except Exception as e:
        print(f"❌ Error building model: {e}")
        raise

    # 5) Training
    print("🎯 Training model...")
    try:
        t_cfg = cfg["training"]
        Trainer = get(t_cfg["block"])
        trainer = Trainer()
        model = trainer.train(model, Xp, y, t_cfg.get("params", {}))
        print("✅ Model training completed")
    except Exception as e:
        print(f"❌ Error in training: {e}")
        raise

    # 6) Evaluation
    print("📈 Evaluating model...")
    try:
        e_cfg = cfg["evaluation"]
        Eval = get(e_cfg["block"])
        evaluator = Eval()
        y_pred = model.predict(Xp)
        metrics = evaluator.evaluate(y, y_pred, e_cfg.get("params", {}))
        
        print("✅ Model evaluation completed")
        print("\n=== 📊 Results ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}" if v == v else f"  {k}: NaN")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        raise

    print("\n🎉 Pipeline execution completed successfully!")
    return {"model": model, "metrics": metrics, "preprocessor": prep}


def validate_pipeline_config(config_path: Path, config_name: str) -> bool:
    """
    Validate that a pipeline configuration has all required components.
    
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
            print(f"❌ Invalid pipeline config. Missing: {', '.join(missing_sections)}")
            return False
            
        print("✅ Pipeline configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating config: {e}")
        return False


def get_pipeline_info(config_path: Path, config_name: str) -> dict:
    """
    Get information about a pipeline configuration.
    
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
            "runtime": cfg.get("runtime", {}).get("block", "unknown")
        }
        
        return info
        
    except Exception as e:
        return {"error": str(e)}
