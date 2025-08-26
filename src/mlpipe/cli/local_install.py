"""
Local installation utilities for hep-ml-templates.
Allows users to download blocks and configs to their project directory.
"""

import shutil
import os
from pathlib import Path
from typing import List, Dict, Set, Optional

# Mapping of extras to their corresponding blocks and configs
EXTRAS_TO_BLOCKS = {
    # Individual components
    'data-csv': {
        'blocks': ['ingest/csv_loader.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['data/csv_demo.yaml']
    },
    'data-higgs': {
        'blocks': ['ingest/csv_loader.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['data/higgs_uci.yaml']
    },
    'model-xgb': {
        'blocks': ['model/xgb_classifier.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/xgb_classifier.yaml']
    },
    'model-decision-tree': {
        'blocks': ['model/decision_tree.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/decision_tree.yaml']
    },
    'model-torch': {
        'blocks': ['model/ae_lightning.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/ae_lightning.yaml']
    },
    'model-gnn': {
        'blocks': ['model/gnn_pyg.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/gnn_pyg.yaml']
    },
    'preprocessing': {
        'blocks': ['preprocessing/standard_scaler.py', 'preprocessing/onehot_encoder.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['preprocessing/standard.yaml']
    },
    'feature-eng': {
        'blocks': ['feature_eng/column_selector.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['feature_eng/demo_features.yaml', 'feature_eng/column_selector.yaml']
    },
    'evaluation': {
        'blocks': ['evaluation/classification_metrics.py', 'evaluation/reconstruction_metrics.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['evaluation/classification.yaml', 'evaluation/reconstruction.yaml']
    },
    
    # Algorithm-specific extras (combinations)
    'xgb': {
        'blocks': ['model/xgb_classifier.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/xgb_classifier.yaml', 'preprocessing/standard.yaml']
    },
    'decision-tree': {
        'blocks': ['model/decision_tree.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/decision_tree.yaml', 'preprocessing/standard.yaml']
    },
    'torch': {
        'blocks': ['model/ae_lightning.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/ae_lightning.yaml', 'preprocessing/standard.yaml']
    },
    'gnn': {
        'blocks': ['model/gnn_pyg.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/gnn_pyg.yaml', 'preprocessing/standard.yaml']
    },
    
    # Complete pipeline bundles
    'pipeline-xgb': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'feature_eng/column_selector.py',
            'model/xgb_classifier.py',
            'evaluation/classification_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],
        'configs': [
            'pipeline.yaml',
            'data/higgs_uci.yaml',
            'data/csv_demo.yaml',
            'preprocessing/standard.yaml',
            'feature_eng/column_selector.yaml',
            'model/xgb_classifier.yaml',
            'evaluation/classification.yaml',
            'runtime/local_cpu.yaml'
        ]
    },
    'pipeline-decision-tree': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'feature_eng/column_selector.py',
            'model/decision_tree.py',
            'evaluation/classification_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],
        'configs': [
            'pipeline.yaml',
            'data/higgs_uci.yaml',
            'data/csv_demo.yaml',
            'preprocessing/standard.yaml',
            'feature_eng/column_selector.yaml',
            'model/decision_tree.yaml',
            'evaluation/classification.yaml',
            'runtime/local_cpu.yaml'
        ]
    },
    'pipeline-torch': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'model/ae_lightning.py',
            'evaluation/reconstruction_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],
        'configs': [
            'pipeline.yaml',
            'data/csv_demo.yaml',
            'preprocessing/standard.yaml',
            'model/ae_lightning.yaml',
            'evaluation/reconstruction.yaml',
            'runtime/local_gpu.yaml'
        ]
    },
    'pipeline-gnn': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'model/gnn_pyg.py',
            'evaluation/classification_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],
        'configs': [
            'pipeline.yaml',
            'data/custom_hep_example.yaml',
            'preprocessing/standard.yaml',
            'model/gnn_pyg.yaml',
            'evaluation/classification.yaml',
            'runtime/local_gpu.yaml'
        ]
    },
    
    # Bundle everything
    'all': {
        'blocks': [
            'ingest/csv_loader.py',
            'ingest/uproot_loader.py',
            'preprocessing/standard_scaler.py',
            'preprocessing/onehot_encoder.py',
            'feature_eng/column_selector.py',
            'model/xgb_classifier.py',
            'model/decision_tree.py',
            'model/ae_lightning.py',
            'model/gnn_pyg.py',
            'evaluation/classification_metrics.py',
            'evaluation/reconstruction_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py', 'metrics.py'],
        'configs': [
            'pipeline.yaml',
            'data/higgs_uci.yaml',
            'data/csv_demo.yaml',
            'data/custom_hep_example.yaml',
            'preprocessing/standard.yaml',
            'feature_eng/column_selector.yaml',
            'feature_eng/demo_features.yaml',
            'model/xgb_classifier.yaml',
            'model/decision_tree.yaml',
            'model/ae_lightning.yaml',
            'model/gnn_pyg.yaml',
            'evaluation/classification.yaml',
            'evaluation/reconstruction.yaml',
            'runtime/local_cpu.yaml',
            'runtime/local_gpu.yaml'
        ]
    }
}

def get_package_path() -> Path:
    """Get the path to the installed hep-ml-templates package."""
    try:
        # Try to get the package path using the mlpipe module
        import mlpipe
        mlpipe_path = Path(mlpipe.__file__).parent  # This is src/mlpipe
        return mlpipe_path
    except ImportError:
        # Fallback: try to find it using importlib
        try:
            import importlib.util
            spec = importlib.util.find_spec('mlpipe')
            if spec and spec.origin:
                return Path(spec.origin).parent
        except:
            pass
        raise FileNotFoundError("Could not locate hep-ml-templates installation")

def get_blocks_and_configs_for_extras(extras: List[str]) -> Dict[str, Set[str]]:
    """
    Given a list of extras, return the blocks, core modules, and configs that should be downloaded.
    
    Args:
        extras: List of extra names (e.g., ['model-xgb', 'data-higgs'])
    
    Returns:
        Dict with 'blocks', 'core', and 'configs' keys containing sets of file paths
    """
    all_blocks = set()
    all_core = set()
    all_configs = set()
    
    # Always include essential core modules needed for CLI functionality
    essential_core = {"registry.py", "interfaces.py", "config.py", "utils.py"}
    
    for extra in extras:
        if extra in EXTRAS_TO_BLOCKS:
            mapping = EXTRAS_TO_BLOCKS[extra]
            all_blocks.update(mapping.get('blocks', []))
            all_core.update(mapping.get('core', []))
            all_configs.update(mapping.get('configs', []))
        else:
            print(f"⚠️  Warning: Unknown extra '{extra}' - skipping")
    
    # Add essential core modules (always needed for CLI to work)
    all_core.update(essential_core)
    
    # If "all" is requested, include everything including metrics.py
    if "all" in extras:
        all_core.add("metrics.py")
    
    return {
        'blocks': all_blocks,
        'core': all_core,
        'configs': all_configs
    }

def copy_core_modules(core_modules: Set[str], source_dir: Path, target_dir: Path):
    """Copy core modules from source to target directory."""
    core_source = source_dir / 'core'  # This is src/mlpipe/core
    
    if not core_source.exists():
        raise FileNotFoundError(f"Core directory not found: {core_source}")
    
    # Create the target core directory
    target_core = target_dir / 'mlpipe' / 'core'
    target_core.mkdir(parents=True, exist_ok=True)
    
    for core_file in core_modules:
        source_file = core_source / core_file
        target_file = target_core / core_file
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied core module: {core_file}")
        else:
            print(f"⚠️  Warning: Core module not found: {source_file}")
    
    # Always copy __init__.py for the core module
    core_init = core_source / '__init__.py'
    target_init = target_core / '__init__.py'
    if core_init.exists():
        shutil.copy2(core_init, target_init)

def copy_blocks(blocks: Set[str], source_dir: Path, target_dir: Path):
    """Copy block files from source to target directory."""
    blocks_source = source_dir / 'blocks'
    
    if not blocks_source.exists():
        raise FileNotFoundError(f"Blocks directory not found: {blocks_source}")
    
    # Keep track of which module categories were installed
    installed_modules = {}
    
    for block_path in blocks:
        source_file = blocks_source / block_path
        target_file = target_dir / 'mlpipe' / 'blocks' / block_path
        
        if source_file.exists():
            # Create target directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied block: {block_path}")
            
            # Track which categories were installed for __init__.py generation
            category = block_path.split('/')[0]  # e.g., 'ingest', 'model', 'preprocessing'
            module_name = Path(block_path).stem  # e.g., 'csv_loader', 'xgb_classifier'
            if category not in installed_modules:
                installed_modules[category] = []
            installed_modules[category].append(module_name)
            
            # Copy category-level __init__.py files
            category_init = blocks_source / category / '__init__.py'
            target_category_init = target_dir / 'mlpipe' / 'blocks' / category / '__init__.py'
            if category_init.exists() and not target_category_init.exists():
                target_category_init.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(category_init, target_category_init)
        else:
            print(f"⚠️  Warning: Block file not found: {source_file}")
    
    # Create custom blocks/__init__.py that only imports installed blocks
    create_custom_blocks_init(installed_modules, target_dir / 'mlpipe' / 'blocks' / '__init__.py')
    
    # Create main mlpipe/__init__.py
    create_main_mlpipe_init(target_dir / 'mlpipe' / '__init__.py')

def create_custom_blocks_init(installed_modules: Dict[str, List[str]], init_file_path: Path):
    """Create a custom __init__.py file for blocks that only imports installed modules."""
    
    init_content = ['# Auto-generated __init__.py for locally installed blocks']
    init_content.append('# Only imports the blocks that were actually installed')
    init_content.append('')
    
    for category, modules in installed_modules.items():
        for module in modules:
            init_content.append(f'try:')
            init_content.append(f'    from .{category} import {module}')
            # Add a comment about what this registers
            if category == 'ingest' and module == 'csv_loader':
                init_content[-1] += '                 # registers "ingest.csv"'
            elif category == 'model' and module == 'xgb_classifier':
                init_content[-1] += '              # registers "model.xgb_classifier"'
            elif category == 'model' and module == 'decision_tree':
                init_content[-1] += '               # registers "model.decision_tree"'
            elif category == 'preprocessing' and module == 'standard_scaler':
                init_content[-1] += '     # registers "preprocessing.standard_scaler"'
            elif category == 'feature_eng' and module == 'column_selector':
                init_content[-1] += '       # registers "feature.column_selector"'
            elif category == 'evaluation' and module == 'classification_metrics':
                init_content[-1] += ' # registers "eval.classification"'
            init_content.append(f'except ImportError:')
            init_content.append(f'    pass  # Optional dependency not available')
            init_content.append('')
    
    # Write the custom __init__.py file
    init_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(init_file_path, 'w') as f:
        f.write('\n'.join(init_content))
    
    print(f"✅ Created custom blocks/__init__.py with {sum(len(modules) for modules in installed_modules.values())} imports")

def create_main_mlpipe_init(init_file_path: Path):
    """Create the main mlpipe/__init__.py file."""
    
    init_content = [
        '"""',
        'HEP ML Templates - Modular ML Pipeline Framework',
        '"""',
        '',
        '__version__ = "0.1.0"',
        '',
        '# Import core modules',
        'try:',
        '    from . import core',
        '    from . import blocks  # This will register all available blocks',
        'except ImportError:',
        '    pass  # Allow partial installations',
        ''
    ]
    
    init_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(init_file_path, 'w') as f:
        f.write('\n'.join(init_content))
    
    print("✅ Created main mlpipe/__init__.py")

def copy_configs(configs: Set[str], source_dir: Path, target_dir: Path):
    """Copy config files from source to target directory."""
    # The config files are in the hep-ml-templates root directory
    # source_dir is src/mlpipe, so we go up two levels to get to hep-ml-templates root
    config_source = source_dir.parent.parent / 'configs'  # src/mlpipe -> src -> hep-ml-templates -> configs
    
    if not config_source.exists():
        raise FileNotFoundError(f"Config directory not found: {config_source}")
    
    for config_path in configs:
        source_file = config_source / config_path
        target_file = target_dir / 'configs' / config_path
        
        if source_file.exists():
            # Create target directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied config: {config_path}")
        else:
            print(f"⚠️  Warning: Config file not found: {source_file}")

def install_local(extras: List[str], target_dir: str) -> bool:
    """
    Install blocks and configs locally based on the provided extras.
    
    Args:
        extras: List of extra names to install
        target_dir: Directory where to install everything
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🚀 Installing hep-ml-templates locally...")
        print(f"📦 Extras: {', '.join(extras)}")
        
        # Resolve target directory
        target_path = Path(target_dir).resolve()
        target_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Installing to: {target_path}")
        
        # Get source directory (installed package)
        package_path = get_package_path()
        print(f"📦 Source package: {package_path}")
        
        # Get blocks and configs to download
        to_download = get_blocks_and_configs_for_extras(extras)
        
        print(f"\n📋 Will install:")
        print(f"   🧩 {len(to_download['blocks'])} blocks")
        print(f"   🔧 {len(to_download['core'])} core modules")
        print(f"   ⚙️  {len(to_download['configs'])} configs")
        
        # Copy core modules first (blocks depend on them)
        if to_download['core']:
            print(f"\n🔧 Installing core modules...")
            copy_core_modules(to_download['core'], package_path, target_path)
        
        # Copy blocks
        if to_download['blocks']:
            print(f"\n🧩 Installing blocks...")
            copy_blocks(to_download['blocks'], package_path, target_path)
        
        # Copy configs  
        if to_download['configs']:
            print(f"\n⚙️  Installing configs...")
            copy_configs(to_download['configs'], package_path, target_path)
        
        # Create a simple setup.py for pip install -e
        create_setup_py(target_path, extras)
        
        # Create a simple CLI script for the local installation
        create_cli_script(target_path)
        
        print(f"\n🎉 Local installation complete!")
        print(f"📁 Files installed in: {target_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. cd {target_path}")
        print(f"   2. pip install -e .")
        print(f"   3. Use: mlpipe run")
        
        return True
        
    except Exception as e:
        print(f"❌ Local installation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_setup_py(target_dir: Path, extras: List[str]):
    """Create a simple setup.py for local installation."""
    
    setup_content = f'''"""
Setup script for locally installed hep-ml-templates components.
Installed extras: {', '.join(extras)}
"""

from setuptools import setup, find_packages

setup(
    name="hep-ml-templates-local",
    version="0.1.0",
    description="Locally installed HEP ML Templates components",
    packages=find_packages(),
    install_requires=[
        "omegaconf>=2.3",
        "pandas>=2.0",
        "numpy>=1.22",
        "scikit-learn>=1.2",
        "hydra-core>=1.3",
    ],
    extras_require={{
        "xgb": ["xgboost>=1.7"],
        "torch": ["torch>=2.0", "pytorch-lightning>=2.0"],
        "gnn": ["torch-geometric>=2.4", "torch>=2.0"],
    }},
    python_requires=">=3.9",
    entry_points={{
        "console_scripts": [
            "mlpipe=mlpipe.cli.main:main",
        ],
    }},
)
'''
    
    setup_file = target_dir / 'setup.py'
    with open(setup_file, 'w') as f:
        f.write(setup_content)
    
    print("✅ Created setup.py for local installation")

def create_cli_script(target_dir: Path):
    """Create a simple CLI script for the locally installed components."""
    
    cli_content = '''#!/usr/bin/env python3
"""
Simple CLI for locally installed hep-ml-templates components.
"""

import sys
from pathlib import Path

# Add the current directory to Python path so we can import mlpipe
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mlpipe.core.registry import list_blocks
    from mlpipe.core.config import load_config
    import mlpipe.blocks  # This will register available blocks
    
    def main():
        if len(sys.argv) < 2:
            print("Usage: python mlpipe_cli.py <command>")
            print("Commands:")
            print("  list-blocks    - List available blocks")
            print("  list-configs   - List available configurations")  
            return
        
        command = sys.argv[1]
        
        if command == "list-blocks":
            print("Available blocks:")
            for name in sorted(list_blocks()):
                print(f"  {name}")
        
        elif command == "list-configs":
            configs_dir = Path("configs")
            if configs_dir.exists():
                print("Available configurations:")
                for config_file in sorted(configs_dir.glob("**/*.yaml")):
                    relative_path = config_file.relative_to(configs_dir)
                    print(f"  {relative_path}")
            else:
                print("No configs directory found")
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: list-blocks, list-configs")

    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing mlpipe components: {e}")
    print("Make sure you've installed the requirements and this directory is in PYTHONPATH")
'''
    
    cli_file = target_dir / 'mlpipe_cli.py'
    with open(cli_file, 'w') as f:
        f.write(cli_content)
    
    # Make it executable
    import stat
    cli_file.chmod(cli_file.stat().st_mode | stat.S_IEXEC)
    
    print("✅ Created mlpipe_cli.py script")
