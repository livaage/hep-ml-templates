from __future__ import annotations
import argparse
from pathlib import Path
from mlpipe.core.registry import list_blocks
from mlpipe.pipelines.xgb_basic.run import run_pipeline
import mlpipe.blocks  # Import to register all blocks

def list_available_configs(config_path: str = "configs"):
    """List available configurations in a helpful format."""
    config_dir = Path(config_path)
    if not config_dir.exists():
        print(f"❌ Configuration directory '{config_path}' not found.")
        return
    
    # List pipeline configs (top-level .yaml files)
    pipeline_configs = list(config_dir.glob("*.yaml"))
    print("📋 Available pipeline configurations:")
    if pipeline_configs:
        for config_file in sorted(pipeline_configs):
            print(f"  • {config_file.stem}")
        print(f"\n🚀 Usage: mlpipe run --config-name <config_name>")
        print(f"   Example: mlpipe run --config-name pipeline")
    else:
        print("   (none found)")
    
    # List modular configs by category
    print(f"\n🧩 Available modular configurations:")
    categories = ["data", "model", "preprocessing", "feature_eng", "training", "evaluation"]
    
    for category in categories:
        cat_dir = config_dir / category
        if cat_dir.exists():
            configs = list(cat_dir.glob("*.yaml"))
            if configs:
                print(f"  📁 {category}:")
                for config_file in sorted(configs):
                    print(f"     • {config_file.stem}")
    
    print(f"\n✨ You can override any component:")
    print(f"   mlpipe run --overrides data=csv_demo model=xgb_classifier")
    print(f"   mlpipe run --overrides data=higgs_uci")

def main():
    parser = argparse.ArgumentParser("mlpipe", 
                                   description="HEP ML Templates - Modular ML Pipeline Framework")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a pipeline")
    p_run.add_argument("--pipeline", default="xgb_basic", 
                       help="Pipeline implementation to use (default: xgb_basic)")
    p_run.add_argument("--config-path", default="configs", 
                       help="Path to configuration directory (default: configs)")
    p_run.add_argument("--config-name", default="pipeline", 
                       help="Pipeline configuration file name without .yaml extension (default: pipeline)")
    p_run.add_argument("--overrides", nargs="*", default=[], 
                       help="Override config values (e.g., data=csv_demo model=xgb_classifier)")

    p_list_blocks = sub.add_parser("list-blocks", help="List available blocks")
    
    p_list_configs = sub.add_parser("list-configs", help="List available configurations")
    p_list_configs.add_argument("--config-path", default="configs")

    args = parser.parse_args()
    
    try:
        if args.cmd == "run":
            run_pipeline(pipeline=args.pipeline, config_path=args.config_path, 
                        config_name=args.config_name, overrides=args.overrides)
        elif args.cmd == "list-blocks":
            print("Available blocks:")
            for name in sorted(list_blocks()):
                print(f"  {name}")
        elif args.cmd == "list-configs":
            list_available_configs(args.config_path)
    except FileNotFoundError as e:
        if ".yaml" in str(e):
            print(f"❌ Error: Configuration file not found")
            print(f"Looking for: {e}")
            print()
            list_available_configs(args.config_path if hasattr(args, 'config_path') else "configs")
        else:
            raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
