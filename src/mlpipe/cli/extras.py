"""Helpers backing the `mlpipe` CLI's extras subcommands.

These functions are imported by `mlpipe.cli.main` to implement `list-extras`,
`validate-extras`, `extra-details`, and `preview-install`. There is no
standalone CLI entry point — everything is reachable via `mlpipe ...`.
"""

import sys

from .local_install import (
    EXTRAS_TO_BLOCKS,
    get_blocks_and_configs_for_extras,
    validate_extras_mappings,
)


def list_extras() -> None:
    """Print all available extras grouped by category."""
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


def show_extra_details(extra_name: str) -> None:
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


def preview_installation(extras: list[str]) -> None:
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
