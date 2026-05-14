from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlpipe.cli.extras import (
    list_extras,
    preview_installation,
    show_extra_details,
    validate_installation,
)
from mlpipe.cli.local_install import install_local
from mlpipe.core.pipeline_generator import generate_pipeline_config, list_available_pipelines
from mlpipe.core.registry import list_blocks
from mlpipe.core.universal_runner import get_pipeline_info, run_pipeline, validate_pipeline_config


def _try_import_local_blocks() -> None:
    """If the user is in a project scaffolded by `mlpipe install-local`, import its
    blocks before the global ones so local edits take precedence.
    """
    cwd = Path.cwd()
    if not (cwd / "mlpipe" / "blocks" / "__init__.py").exists():
        return

    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    try:
        import importlib

        import mlpipe.blocks as local_blocks  # noqa: F401

        importlib.reload(local_blocks)
    except Exception:  # noqa: S110 — fall back to global blocks if the local copy is incomplete
        pass


def list_available_configs(config_path: str = "configs") -> None:
    config_dir = Path(config_path)
    if not config_dir.exists():
        print(f"Error: configuration directory {config_path!r} not found", file=sys.stderr)
        return

    pipeline_configs = sorted(config_dir.glob("*.yaml"))
    print("Available pipeline configurations:")
    if pipeline_configs:
        for config_file in pipeline_configs:
            print(f"  {config_file.stem}")
        print("\nUsage: mlpipe run --config-name <config_name>")
    else:
        print("  (none found)")

    print("\nAvailable modular configurations:")
    for category in ("data", "model", "preprocessing", "feature_eng", "training", "evaluation"):
        cat_dir = config_dir / category
        if not cat_dir.exists():
            continue
        configs = sorted(cat_dir.glob("*.yaml"))
        if not configs:
            continue
        print(f"  {category}:")
        for config_file in configs:
            print(f"    {config_file.stem}")

    print("\nOverride any component on the command line, e.g.:")
    print("  mlpipe run --overrides data=csv_demo model=xgb_classifier")


def main() -> None:
    # Local blocks first, then fall back to global blocks for anything still missing.
    _try_import_local_blocks()
    if not (Path.cwd() / "mlpipe" / "blocks" / "__init__.py").exists():
        import mlpipe.blocks  # noqa: F401

    parser = argparse.ArgumentParser(
        "mlpipe", description="hep-ml-templates: modular ML pipeline framework"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a pipeline")
    p_run.add_argument(
        "--pipeline",
        default="auto",
        help="Pipeline implementation (default: auto - determined from config)",
    )
    p_run.add_argument(
        "--config-path",
        default="configs",
        help="Path to configuration directory (default: configs)",
    )
    p_run.add_argument(
        "--config-name",
        default="pipeline",
        help="Pipeline configuration file name without .yaml extension (default: pipeline)",
    )
    p_run.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Override config values (e.g., data=csv_demo model=xgb_classifier)",
    )

    sub.add_parser("list-blocks", help="List available blocks")

    p_list_configs = sub.add_parser("list-configs", help="List available configurations")
    p_list_configs.add_argument("--config-path", default="configs")

    p_generate = sub.add_parser("generate-pipeline", help="Generate a pipeline configuration")
    p_generate.add_argument(
        "pipeline_type",
        choices=["decision-tree", "xgb", "neural", "torch", "gnn"],
        help="Type of pipeline to generate",
    )
    p_generate.add_argument(
        "--output", default="pipeline.yaml", help="Output file path (default: pipeline.yaml)"
    )

    p_validate = sub.add_parser("validate-config", help="Validate a pipeline configuration")
    p_validate.add_argument("--config-path", default="configs")
    p_validate.add_argument("--config-name", default="pipeline")

    p_info = sub.add_parser("pipeline-info", help="Show information about a pipeline configuration")
    p_info.add_argument("--config-path", default="configs")
    p_info.add_argument("--config-name", default="pipeline")

    sub.add_parser("list-pipeline-templates", help="List available pipeline templates")

    p_install = sub.add_parser(
        "install-local", help="Install blocks and configs locally to your project"
    )
    p_install.add_argument(
        "extras",
        nargs="+",
        help="Extras to install locally (e.g., model-xgb data-higgs pipeline-xgb all)",
    )
    p_install.add_argument(
        "--target-dir", required=True, help="Directory where to install the local components"
    )

    sub.add_parser("list-extras", help="List all available extras")
    sub.add_parser("validate-extras", help="Validate extras configuration")

    p_extra_details = sub.add_parser("extra-details", help="Show details for a specific extra")
    p_extra_details.add_argument("extra", help="Name of the extra to show details for")

    p_preview_install = sub.add_parser("preview-install", help="Preview what would be installed")
    p_preview_install.add_argument("extras", nargs="+", help="Extras to preview")

    args = parser.parse_args()

    try:
        if args.cmd == "run":
            run_pipeline(
                pipeline=args.pipeline,
                config_path=args.config_path,
                config_name=args.config_name,
                overrides=args.overrides,
            )
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
            print("Pipeline configuration info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        elif args.cmd == "list-pipeline-templates":
            pipelines = list_available_pipelines()
            print("Available pipeline templates:")
            for name, info in pipelines.items():
                print(f"  {name}: {info['description']}")
                print(f"    Model: {info['config']['model']}")
                deps = ", ".join(info["dependencies"]["required"])
                print(f"    Dependencies: {deps}")
                print()
        elif args.cmd == "install-local":
            if not install_local(args.extras, args.target_dir):
                sys.exit(1)
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
            print(f"Error: configuration file not found: {e}", file=sys.stderr)
            list_available_configs(getattr(args, "config_path", "configs"))
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
