#!/usr/bin/env python3
"""Command-line utility for inspecting and installing hep-ml-templates extras."""

import argparse
import sys

from .local_install import (
    EXTRAS_TO_BLOCKS,
    get_blocks_and_configs_for_extras,
    install_local,
    validate_extras_mappings,
)


def list_extras():
    """List all available extras grouped by category."""
    print("Available extras:")

    groups: dict[str, list[tuple[str, int, int]]] = {
        "Complete pipelines": [],
        "Individual models": [],
        "Algorithm combos (model + preprocessing)": [],
        "Component categories": [],
        "Data sources": [],
        "Special": [],
    }

    for name in sorted(EXTRAS_TO_BLOCKS):
        mapping = EXTRAS_TO_BLOCKS[name]
        entry = (name, len(mapping.get("blocks", [])), len(mapping.get("configs", [])))
        if name.startswith("data-"):
            groups["Data sources"].append(entry)
        elif name.startswith("model-"):
            groups["Individual models"].append(entry)
        elif name.startswith("pipeline-"):
            groups["Complete pipelines"].append(entry)
        elif name in {"preprocessing", "feature-eng", "evaluation"}:
            groups["Component categories"].append(entry)
        elif name == "all":
            groups["Special"].append(entry)
        else:
            groups["Algorithm combos (model + preprocessing)"].append(entry)

    for title, entries in groups.items():
        if not entries:
            continue
        print(f"\n{title}:")
        for name, blocks, configs in entries:
            print(f"  {name:<25} ({blocks} blocks, {configs} configs)")


def validate_installation() -> bool:
    """Validate the extras → blocks/configs mappings; return True if all clean."""
    issues = validate_extras_mappings()
    if not any(issues.values()):
        print("All extras configurations are valid.")
        return True

    print("Configuration issues found:", file=sys.stderr)
    for issue_type, issue_list in issues.items():
        if not issue_list:
            continue
        print(f"\n{issue_type.replace('_', ' ').title()}:", file=sys.stderr)
        for issue in issue_list:
            print(f"  - {issue}", file=sys.stderr)
    return False


def show_extra_details(extra_name: str):
    """Show what blocks, core modules, configs, and data an extra pulls in."""
    if extra_name not in EXTRAS_TO_BLOCKS:
        print(f"Error: unknown extra {extra_name!r}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(EXTRAS_TO_BLOCKS))}", file=sys.stderr)
        return

    mapping = EXTRAS_TO_BLOCKS[extra_name]
    print(f"Details for '{extra_name}':")

    for label, key in (
        ("Blocks", "blocks"),
        ("Core modules", "core"),
        ("Configurations", "configs"),
        ("Data files", "data"),
    ):
        items = mapping.get(key, [])
        if not items:
            continue
        print(f"\n{label} ({len(items)}):")
        for item in items:
            print(f"  - {item}")


def preview_installation(extras: list[str]):
    """Preview what `install_local(extras)` would copy without writing anything."""
    print(f"Installation preview for: {', '.join(extras)}")

    to_install = get_blocks_and_configs_for_extras(extras)
    for label, key in (
        ("Blocks", "blocks"),
        ("Core modules", "core"),
        ("Configurations", "configs"),
        ("Data files", "data"),
    ):
        items = sorted(to_install.get(key, []))
        if not items:
            continue
        print(f"\n{label} ({len(items)}):")
        for item in items:
            print(f"  - {item}")


def main():
    parser = argparse.ArgumentParser(
        description="hep-ml-templates extras manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlpipe-manager list
  mlpipe-manager validate
  mlpipe-manager details model-xgb
  mlpipe-manager preview model-xgb preprocessing
  mlpipe-manager install model-xgb ./my-project
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("list", help="List all available extras")
    subparsers.add_parser("validate", help="Validate extras configuration")

    details_parser = subparsers.add_parser("details", help="Show details for a specific extra")
    details_parser.add_argument("extra", help="Name of the extra to show details for")

    preview_parser = subparsers.add_parser("preview", help="Preview what would be installed")
    preview_parser.add_argument("extras", nargs="+", help="Extras to preview")

    install_parser = subparsers.add_parser("install", help="Install extras to a directory")
    install_parser.add_argument("extras", nargs="+", help="Extras to install")
    install_parser.add_argument("directory", help="Target directory for installation")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    if args.command == "list":
        list_extras()
    elif args.command == "validate":
        validate_installation()
    elif args.command == "details":
        show_extra_details(args.extra)
    elif args.command == "preview":
        preview_installation(args.extras)
    elif args.command == "install":
        print(f"Installing extras: {', '.join(args.extras)}")
        print(f"Target directory: {args.directory}")
        if not install_local(args.extras, args.directory):
            print("Installation failed", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
