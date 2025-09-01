"""
Graph CSV Loader - Loads CSV data for graph neural network training.

This loader loads tabular CSV data and prepares it for graph neural network
processing. The graph structure creation happens in the model block.
"""

from typing import Any, Dict, Tuple
import pandas as pd

from mlpipe.core.interfaces import DataIngestor
from mlpipe.core.registry import register


@register("ingest.graph_csv")
class GraphCSVLoader(DataIngestor):
    """
    Load CSV data for graph neural network training.
    
    This loader loads tabular CSV data and prepares it for graph neural network
    processing. The graph structure creation happens in the model block.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Graph CSV Loader.
        
        Args:
            config: Configuration dictionary with keys:
                - file_path: Path to CSV file
                - target_column: Name of target column
                - edge_threshold: Similarity threshold for creating edges (0-1)
                - max_edges_per_node: Maximum number of edges per node
                - has_header: Whether CSV has header row
                - separator: CSV separator
                - encoding: File encoding
        """
        if config is None:
            config = {}
            
        self.file_path = config.get('file_path', '')
        self.target_column = config.get('target_column', 'target')
        self.edge_threshold = config.get('edge_threshold', 0.3)
        self.max_edges_per_node = config.get('max_edges_per_node', 8)
        self.has_header = config.get('has_header', True)
        self.separator = config.get('separator', ',')
        self.encoding = config.get('encoding', 'utf-8')
        
        print("🔗 Graph CSV Loader")
        print("========================================")
        
    def load(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Load CSV data for graph processing.
        
        Returns:
            Tuple of (features_dataframe, target_series, metadata) 
            Note: Graph creation happens in the model block
        """
        # Load CSV data
        df = pd.read_csv(self.file_path, 
                        header=0 if self.has_header else None,
                        sep=self.separator, 
                        encoding=self.encoding)
        
        print(f"📁 Loading CSV from: {self.file_path}")
        print(f"📊 Dataset structure detected:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Has header: {self.has_header}")
        
        # Separate features and target
        target = df[self.target_column]
        features = df.drop(columns=[self.target_column])
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Features shape: {features.shape}")
        print(f"   - Target shape: {target.shape}")
        print(f"   - Target classes: {target.nunique()}")
        
        # Create metadata for pipeline 
        metadata = {
            "loader_type": "graph_csv",
            "file_path": self.file_path,
            "target_column": self.target_column,
            "dataset_shape": df.shape,
            "feature_columns": list(features.columns),
            "num_classes": target.nunique()
        }
        
        return features, target, metadata
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the CSV dataset."""
        return {
            "loader_type": "graph_csv",
            "file_path": self.file_path,
            "target_column": self.target_column
        }
