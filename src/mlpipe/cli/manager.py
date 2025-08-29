#!/usr/bin/env python3
"""
HEP-ML-Templates Extras Manager
Command-line utility for managing local installations and extras.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from .local_install import (
    EXTRAS_TO_BLOCKS,
    validate_extras_mappings,
    install_local,
    get_blocks_and_configs_for_extras
)


def list_extras():
    """List all available extras with their descriptions."""
    print("📦 Available HEP-ML-Templates Extras:")
    print("=" * 50)

    # Group extras by type
    data_extras = []
    model_extras = []
    algorithm_combos = []
    category_extras = []
    pipeline_extras = []
    special_extras = []

    for name in sorted(EXTRAS_TO_BLOCKS.keys()):
        mapping = EXTRAS_TO_BLOCKS[name]
        block_count = len(mapping.get('blocks', []))
        config_count = len(mapping.get('configs', []))

        if name.startswith('data-'):
            data_extras.append((name, block_count, config_count))
        elif name.startswith('model-'):
            model_extras.append((name, block_count, config_count))
        elif name.startswith('pipeline-'):
            pipeline_extras.append((name, block_count, config_count))
        elif name in ['preprocessing', 'feature-eng', 'evaluation']:
            category_extras.append((name, block_count, config_count))
        elif name == 'all':
            special_extras.append((name, block_count, config_count))
        else:
            algorithm_combos.append((name, block_count, config_count))

    def print_group(title: str, extras_list: List[Tuple[str, int, int]]):
        if extras_list:
            print(f"\n{title}:")
            for name, blocks, configs in extras_list:
                print(f"  {name:<25} ({blocks} blocks, {configs} configs)")

    print_group("🎯 Complete Pipelines", pipeline_extras)
    print_group("🧠 Individual Models", model_extras)
    print_group("⚡ Algorithm Combos (Model + Preprocessing)", algorithm_combos)
    print_group("🏗️  Component Categories", category_extras)
    print_group("📊 Data Sources", data_extras)
    print_group("🌟 Special", special_extras)


def validate_installation():
    """Validate the current extras configuration."""
    print("🔍 Validating extras configuration...")
    issues = validate_extras_mappings()

    if not any(issues.values()):
        print("✅ All extras configurations are valid!")
        return True
    else:
        print("⚠️  Configuration issues found:")

        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n❌ {issue_type.replace('_', ' ').title()}:")
                for issue in issue_list:
                    print(f"  - {issue}")
        return False


def show_extra_details(extra_name: str):
    """Show detailed information about a specific extra."""
    if extra_name not in EXTRAS_TO_BLOCKS:
        print(f"❌ Unknown extra: {extra_name}")
        print(f"Available extras: {', '.join(sorted(EXTRAS_TO_BLOCKS.keys()))}")
        return

    mapping = EXTRAS_TO_BLOCKS[extra_name]

    print(f"📋 Details for '{extra_name}':")
    print("=" * 40)

    print(f"\n🧩 Blocks ({len(mapping.get('blocks', []))}):")
    for block in mapping.get('blocks', []):
        print(f"  - {block}")

    print(f"\n🔧 Core modules ({len(mapping.get('core', []))}):")
    for core in mapping.get('core', []):
        print(f"  - {core}")

    print(f"\n⚙️  Configurations ({len(mapping.get('configs', []))}):")
    for config in mapping.get('configs', []):
        print(f"  - {config}")

    if mapping.get('data'):
        print(f"\n📊 Data files ({len(mapping['data'])}):")
        for data_file in mapping['data']:
            print(f"  - {data_file}")


def preview_installation(extras: List[str]):
    """Preview what would be installed for the given extras."""
    print(f"🔍 Installation preview for: {', '.join(extras)}")
    print("=" * 50)

    to_install = get_blocks_and_configs_for_extras(extras)

    print(f"\n🧩 Blocks to install ({len(to_install['blocks'])}):")
    for block in sorted(to_install['blocks']):
        print(f"  - {block}")

    print(f"\n🔧 Core modules to install ({len(to_install['core'])}):")
    for core in sorted(to_install['core']):
        print(f"  - {core}")

    print(f"\n⚙️  Configurations to install ({len(to_install['configs'])}):")
    for config in sorted(to_install['configs']):
        print(f"  - {config}")

    if to_install['data']:
        print(f"\n📊 Data files to install ({len(to_install['data'])}):")
        for data_file in sorted(to_install['data']):
            print(f"  - {data_file}")


def main():
    """Main entry point for the extras manager CLI."""
    parser = argparse.ArgumentParser(
        description="HEP-ML-Templates Extras Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlpipe-manager list                          # List all available extras
  mlpipe-manager validate                      # Validate extras configuration
  mlpipe-manager details model-xgb             # Show details for specific extra
  mlpipe-manager preview model-xgb preprocessing  # Preview installation
  mlpipe-manager install model-xgb ./my-project   # Install extras to directory
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    subparsers.add_parser('list', help='List all available extras')

    # Validate command
    subparsers.add_parser('validate', help='Validate extras configuration')

    # Details command
    details_parser = subparsers.add_parser('details', help='Show details for a specific extra')
    details_parser.add_argument('extra', help='Name of the extra to show details for')

    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview what would be installed')
    preview_parser.add_argument('extras', nargs='+', help='Extras to preview')

    # Install command
    install_parser = subparsers.add_parser('install', help='Install extras to a directory')
    install_parser.add_argument('extras', nargs='+', help='Extras to install')
    install_parser.add_argument('directory', help='Target directory for installation')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == 'list':
        list_extras()
    elif args.command == 'validate':
        validate_installation()
    elif args.command == 'details':
        show_extra_details(args.extra)
    elif args.command == 'preview':
        preview_installation(args.extras)
    elif args.command == 'install':
        print(f"🚀 Installing extras: {', '.join(args.extras)}")
        print(f"📁 Target directory: {args.directory}")
        success = install_local(args.extras, args.directory)
        if success:
            print("✅ Installation completed successfully!")
        else:
            print("❌ Installation failed!")
            sys.exit(1)


if __name__ == '__main__':
    main()
