"""
ROOT file data loading for High Energy Physics.

This module provides functionality to load data from ROOT files using uproot,
which is commonly used in particle physics data analysis.

Supports:
- Automatic tree detection
- Branch selection and filtering  
- Selection cuts in ROOT syntax
- Jagged array handling
- Large file processing with memory management
"""

try:
    import uproot
    import awkward as ak
    import numpy as np
    import pandas as pd
    UPROOT_AVAILABLE = True
except ImportError:
    UPROOT_AVAILABLE = False

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from mlpipe.core.interfaces import DataIngestor
from mlpipe.core.registry import register


@register("ingest.uproot_loader")
class UprootDataBlock(DataIngestor):
    """
    ROOT file data loader using uproot.
    
    Designed for High Energy Physics data stored in ROOT format.
    Supports automatic tree detection, branch selection, and ROOT-style cuts.
    
    Example usage:
        loader = UprootDataBlock()
        loader.build({
            'file_path': 'data/events.root',
            'tree_name': 'Events',
            'branches': ['pt', 'eta', 'phi', 'mass'],
            'target_branch': 'label',
            'selection_cuts': 'pt > 20 && abs(eta) < 2.5',
            'max_entries': 100000
        })
        X, y = loader.load()
    """
    
    def __init__(self):
        if not UPROOT_AVAILABLE:
            raise ImportError(
                "uproot is required for ROOT file loading. "
                "Install with: pip install uproot awkward"
            )
        super().__init__()
        self.config = {}
        
    def build(self, config: Dict[str, Any]) -> None:
        """Configure the ROOT file loader."""
        default_config = {
            'file_path': None,
            'tree_name': None,  # Auto-detect if None
            'branches': None,   # Load all branches if None
            'target_branch': None,
            'selection_cuts': None,  # ROOT-style selection cuts
            'max_entries': None,     # Load all entries if None
            'flatten_arrays': True,  # Convert jagged arrays to flat
            'verbose': True,
            'chunk_size': 100000,    # For memory management
            'library': 'np'          # 'np' for numpy, 'pd' for pandas
        }
        
        self.config = {**default_config, **config}
        
        # Validate required parameters
        if not self.config['file_path']:
            raise ValueError("file_path is required")
            
        # Check if file exists
        file_path = Path(self.config['file_path'])
        if not file_path.exists():
            raise FileNotFoundError(f"ROOT file not found: {file_path}")
            
        if self.config['verbose']:
            print(f"🔬 ROOT File Loader Configuration:")
            print(f"   File: {self.config['file_path']}")
            print(f"   Tree: {self.config['tree_name'] or 'auto-detect'}")
            print(f"   Branches: {self.config['branches'] or 'all'}")
            print(f"   Target: {self.config['target_branch'] or 'none'}")
            if self.config['selection_cuts']:
                print(f"   Cuts: {self.config['selection_cuts']}")
                
    def _detect_tree(self, file: uproot.ReadOnlyDirectory) -> str:
        """Auto-detect the main tree in the ROOT file."""
        keys = file.keys()
        
        # Filter for TTree objects
        trees = [key for key in keys if hasattr(file[key], 'arrays')]
        
        if not trees:
            raise ValueError("No trees found in ROOT file")
            
        if len(trees) == 1:
            tree_name = trees[0]
        else:
            # Prefer common tree names
            common_names = ['Events', 'tree', 'ntuple', 'data']
            for name in common_names:
                matches = [t for t in trees if name.lower() in t.lower()]
                if matches:
                    tree_name = matches[0]
                    break
            else:
                # Use the largest tree
                tree_sizes = {name: file[name].num_entries for name in trees}
                tree_name = max(tree_sizes.keys(), key=lambda k: tree_sizes[k])
                
        if self.config['verbose']:
            print(f"📊 Auto-detected tree: {tree_name}")
            print(f"   Available trees: {trees}")
            
        return tree_name
        
    def _get_branches(self, tree: uproot.TTree) -> List[str]:
        """Get list of branches to load."""
        all_branches = list(tree.keys())
        
        if self.config['branches'] is None:
            # Load all branches except target
            branches = [b for b in all_branches if b != self.config['target_branch']]
        else:
            # Use specified branches
            branches = self.config['branches']
            # Validate branches exist
            missing = [b for b in branches if b not in all_branches]
            if missing:
                raise ValueError(f"Branches not found: {missing}")
                
        if self.config['verbose']:
            print(f"🌿 Loading {len(branches)} branches:")
            print(f"   {branches[:5]}{'...' if len(branches) > 5 else ''}")
            
        return branches
        
    def _apply_cuts(self, arrays: Dict, tree: uproot.TTree) -> np.ndarray:
        """Apply ROOT-style selection cuts and return boolean mask."""
        if not self.config['selection_cuts']:
            return np.ones(tree.num_entries, dtype=bool)
            
        # Simple cut parsing (extend as needed)
        cuts_str = self.config['selection_cuts']
        
        if self.config['verbose']:
            print(f"✂️  Applying cuts: {cuts_str}")
            
        # For now, support basic cuts like "pt > 20", "abs(eta) < 2.5"
        # This is a simplified implementation - real ROOT cuts are more complex
        try:
            # Replace ROOT functions with numpy equivalents
            cuts_str = cuts_str.replace('abs(', 'np.abs(')
            cuts_str = cuts_str.replace('&&', '&')
            cuts_str = cuts_str.replace('||', '|')
            
            # Build local namespace with arrays
            namespace = {'np': np, **arrays}
            
            # Evaluate the cut expression
            mask = eval(cuts_str, namespace)
            
            if isinstance(mask, (list, tuple)):
                mask = np.array(mask)
                
            return mask.astype(bool)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not apply cuts '{cuts_str}': {e}")
            return np.ones(len(list(arrays.values())[0]), dtype=bool)
            
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from ROOT file."""
        if self.config['verbose']:
            print(f"🚀 Loading ROOT file: {self.config['file_path']}")
            
        with uproot.open(self.config['file_path']) as file:
            # Detect or use specified tree
            tree_name = self.config['tree_name'] or self._detect_tree(file)
            tree = file[tree_name]
            
            if self.config['verbose']:
                print(f"📈 Tree info:")
                print(f"   Entries: {tree.num_entries:,}")
                print(f"   Branches: {len(tree.keys())}")
                
            # Get branches to load
            feature_branches = self._get_branches(tree)
            
            # Load target if specified
            target_data = None
            if self.config['target_branch']:
                if self.config['target_branch'] not in tree.keys():
                    raise ValueError(f"Target branch '{self.config['target_branch']}' not found")
                feature_branches.append(self.config['target_branch'])
                
            # Load data in chunks if needed
            max_entries = self.config['max_entries'] or tree.num_entries
            chunk_size = min(self.config['chunk_size'], max_entries)
            
            all_data = []
            entries_loaded = 0
            
            for start in range(0, max_entries, chunk_size):
                end = min(start + chunk_size, max_entries)
                
                if self.config['verbose'] and max_entries > chunk_size:
                    print(f"📦 Loading chunk {start:,}-{end:,}")
                    
                # Load arrays
                arrays = tree.arrays(
                    feature_branches,
                    entry_start=start,
                    entry_stop=end,
                    library=self.config['library']
                )
                
                # Convert awkward arrays to numpy if needed
                if self.config['flatten_arrays']:
                    for branch in arrays.keys():
                        if hasattr(arrays[branch], 'to_numpy'):
                            arrays[branch] = arrays[branch].to_numpy()
                        elif isinstance(arrays[branch], ak.Array):
                            # Handle jagged arrays
                            try:
                                arrays[branch] = ak.to_numpy(arrays[branch])
                            except ValueError:
                                # If jagged, flatten or take first element
                                arrays[branch] = ak.flatten(arrays[branch]).to_numpy()
                                
                all_data.append(arrays)
                entries_loaded += (end - start)
                
            if self.config['verbose']:
                print(f"✅ Loaded {entries_loaded:,} entries")
                
            # Combine chunks
            if len(all_data) == 1:
                combined_arrays = all_data[0]
            else:
                combined_arrays = {}
                for branch in feature_branches:
                    combined_arrays[branch] = np.concatenate([
                        chunk[branch] for chunk in all_data
                    ])
                    
            # Apply selection cuts
            if self.config['selection_cuts']:
                mask = self._apply_cuts(combined_arrays, tree)
                for branch in combined_arrays.keys():
                    combined_arrays[branch] = combined_arrays[branch][mask]
                    
                if self.config['verbose']:
                    print(f"✂️  After cuts: {mask.sum():,} entries ({100*mask.sum()/len(mask):.1f}%)")
                    
            # Separate features and target
            if self.config['target_branch']:
                target_data = combined_arrays.pop(self.config['target_branch'])
                
            # Convert to arrays
            if isinstance(combined_arrays, dict):
                feature_names = [b for b in feature_branches if b != self.config['target_branch']]
                X = np.column_stack([combined_arrays[branch] for branch in feature_names])
            else:
                X = combined_arrays
                
            y = target_data if target_data is not None else np.array([])
            
            if self.config['verbose']:
                print(f"📊 Final dataset:")
                print(f"   Features: {X.shape}")
                print(f"   Target: {y.shape if len(y) > 0 else 'None'}")
                
            return X, y
