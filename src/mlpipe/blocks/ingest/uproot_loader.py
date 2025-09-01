"""ROOT file data loading for High Energy Physics.

This module provides functionality to load data from ROOT files using uproot.
Only registers the UprootDataBlock if uproot dependencies are available.
"""

# Check for uproot availability before doing anything
try:
    import awkward as ak
    import numpy as np
    import pandas as pd
    import uproot

    UPROOT_AVAILABLE = True
except ImportError:
    UPROOT_AVAILABLE = False

# Only proceed with class definition if uproot is available
if UPROOT_AVAILABLE:
    from pathlib import Path
    from typing import Any, Dict, Optional, Tuple

    from mlpipe.core.interfaces import DataIngestor
    from mlpipe.core.registry import register

    @register("ingest.uproot_loader")
    class UprootDataBlock(DataIngestor):
        """ROOT file data loader using uproot.

        Designed for High Energy Physics data stored in ROOT format.
        """

        def __init__(self, config: Optional[Dict[str, Any]] = None):
            super().__init__(config or {})
            self.file_path = None
            self.tree_name = None
            self.branches = None
            self.target_column = None

        def configure(self, config: Dict[str, Any]) -> "UprootDataBlock":
            """Configure the loader with parameters."""
            self.file_path = config.get("file_path", config.get("path"))
            self.tree_name = config.get("tree_name", config.get("tree", "Events"))
            self.branches = config.get("branches", config.get("columns"))
            self.target_column = config.get("target_column", config.get("label"))
            return self

        def load(self) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
            """Load data from ROOT file."""
            if not self.file_path:
                raise ValueError("file_path must be specified")

            file_path = Path(self.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"ROOT file not found: {file_path}")

            print(f"Loading ROOT file: {file_path}")

            with uproot.open(file_path) as file:
                tree = file[self.tree_name]

                # Load branches
                branch_names = self.branches or list(tree.keys())[:10]  # Limit for safety
                arrays = tree.arrays(branch_names, library="pd")

                # Separate features and target
                if self.target_column and self.target_column in arrays.columns:
                    y = arrays[self.target_column]
                    X = arrays.drop(columns=[self.target_column])
                else:
                    X = arrays
                    y = None

                metadata = {
                    "source_file": str(file_path),
                    "tree_name": self.tree_name,
                    "n_entries": len(X),
                    "n_features": len(X.columns),
                }

                print(f"Loaded {len(X)} entries with {len(X.columns)} features")
                return X, y, metadata

else:
    # If uproot is not available, define a dummy class for informative errors
    class UprootDataBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "UprootDataBlock requires uproot and awkward. "
                "Install with: pip install 'hep-ml-templates[data-uproot]'"
            )
