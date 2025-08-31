from __future__ import annotations
import argparse
from pathlib import Path
from mlpipe.core.registry import list_blocks
from mlpipe.core.universal_runner import run_pipeline, validate_pipeline_config, get_pipeline_info
from mlpipe.core.pipeline_generator import generate_pipeline_config, list_available_pipelines
from mlpipe.cli.local_install import install_local
from mlpipe.cli.manager import (
    list_extras, validate_installation, show_extra_details, preview_installation
)
import mlpipe.blocks  # Import to register all blocks  # noqa: F401


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
        print("\n🚀 Usage: mlpipe run --config-name <config_name>")
        print("   Example: mlpipe run --config-name pipeline")
    else:
        print("   (none found)")

    # List modular configs by category
    print("\n🧩 Available modular configurations:")
    categories = ["data", "model", "preprocessing", "feature_eng", "training", "evaluation"]

    for category in categories:
        cat_dir = config_dir / category
        if cat_dir.exists():
            configs = list(cat_dir.glob("*.yaml"))
            if configs:
                print(f"  📁 {category}:")
                for config_file in sorted(configs):
                    print(f"     • {config_file.stem}")

    print("\n✨ You can override any component:")
    print("   mlpipe run --overrides data=csv_demo model=xgb_classifier")
    print("   mlpipe run --overrides data=higgs_uci")


def main():
    parser = argparse.ArgumentParser("mlpipe",
                                     description="HEP ML Templates - Modular ML Pipeline Framework")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a pipeline")
    p_run.add_argument("--pipeline", default="auto",
                       help="Pipeline implementation (default: auto - determined from config)")
    p_run.add_argument("--config-path", default="configs",
                       help="Path to configuration directory (default: configs)")
    p_run.add_argument("--config-name", default="pipeline",
                       help="Pipeline configuration file name without .yaml "
                       "extension (default: pipeline)")
    p_run.add_argument("--overrides", nargs="*", default=[],
                       help="Override config values (e.g., data=csv_demo model=xgb_classifier)")

    # Add subcommands
    sub.add_parser("list-blocks", help="List available blocks")

    p_list_configs = sub.add_parser("list-configs", help="List available configurations")
    p_list_configs.add_argument("--config-path", default="configs")
    
    # Add pipeline generation command
    p_generate = sub.add_parser("generate-pipeline", help="Generate a pipeline configuration")
    p_generate.add_argument("pipeline_type", choices=["decision-tree", "xgb", "neural", "torch", "gnn"],
                           help="Type of pipeline to generate")
    p_generate.add_argument("--output", default="pipeline.yaml",
                           help="Output file path (default: pipeline.yaml)")
    
    # Add pipeline validation
    p_validate = sub.add_parser("validate-config", help="Validate a pipeline configuration")
    p_validate.add_argument("--config-path", default="configs")
    p_validate.add_argument("--config-name", default="pipeline")
    
    # Add pipeline info
    p_info = sub.add_parser("pipeline-info", help="Show information about a pipeline configuration")
    p_info.add_argument("--config-path", default="configs")
    p_info.add_argument("--config-name", default="pipeline")
    
    # List available pipeline templates
    sub.add_parser("list-pipeline-templates", help="List available pipeline templates")
    
    # Add local installation command
    p_install = sub.add_parser("install-local", help="Install blocks and configs locally to your project")
    p_install.add_argument("extras", nargs="+", 
                          help="Extras to install locally (e.g., model-xgb data-higgs pipeline-xgb all)")
    p_install.add_argument("--target-dir", required=True, 
                          help="Directory where to install the local components")

    # Add extras management commands
    p_list_extras = sub.add_parser("list-extras", help="List all available extras")
    
    p_validate_extras = sub.add_parser("validate-extras", help="Validate extras configuration")
    
    p_extra_details = sub.add_parser("extra-details", help="Show details for a specific extra")
    p_extra_details.add_argument("extra", help="Name of the extra to show details for")
    
    p_preview_install = sub.add_parser("preview-install", help="Preview what would be installed")
    p_preview_install.add_argument("extras", nargs="+", help="Extras to preview")

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
        elif args.cmd == "generate-pipeline":
            output_path = Path(args.output)
            config = generate_pipeline_config(args.pipeline_type, output_path=output_path)
            print(f"Generated {args.pipeline_type} pipeline configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        elif args.cmd == "validate-config":
            validate_pipeline_config(Path(args.config_path), args.config_name)
        elif args.cmd == "pipeline-info":
            info = get_pipeline_info(Path(args.config_path), args.config_name)
            print("Pipeline Configuration Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        elif args.cmd == "list-pipeline-templates":
            pipelines = list_available_pipelines()
            print("Available pipeline templates:")
            for name, info in pipelines.items():
                print(f"  {name}: {info['description']}")
                print(f"    Model: {info['config']['model']}")
                deps = ", ".join(info['dependencies']['required'])
                print(f"    Dependencies: {deps}")
                print()
        elif args.cmd == "install-local":
            success = install_local(args.extras, args.target_dir)
            if not success:
                exit(1)
        elif args.cmd == "list-extras":
            list_extras()
        elif args.cmd == "validate-extras":
            validate_installation()
        elif args.cmd == "extra-details":
            show_extra_details(args.extra)
        elif args.cmd == "preview-install":
            preview_installation(args.extras)
    except FileNotFoundError as e:
        if ".yaml" in str(e):
            print("❌ Error: Configuration file not found")
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
