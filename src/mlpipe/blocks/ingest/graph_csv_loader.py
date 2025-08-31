"""
Graph CSV Loader - Converts tabular data to graph format for GNN training.

This loader takes tabular CSV data and creates a graph structure suitable for
Graph Neural Networks by connecting nodes based on feature similarity.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from mlpipe.core.interfaces import DataBlock
from mlpipe.core.registry import register


@register("ingest.graph_csv")
class GraphCSVLoader(DataBlock):
    """
    Load CSV data and convert to graph format for GNN training.
    
    Creates edges between nodes based on feature similarity, making it
    suitable for node classification tasks with GNNs.
    """
    
    def __init__(self, file_path: str, target_column: str, 
                 edge_threshold: float = 0.5, max_edges_per_node: int = 10,
                 **kwargs):
        """
        Initialize Graph CSV Loader.
        
        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            edge_threshold: Similarity threshold for creating edges (0-1)
            max_edges_per_node: Maximum number of edges per node
        """
        self.file_path = file_path
        self.target_column = target_column
        self.edge_threshold = edge_threshold
        self.max_edges_per_node = max_edges_per_node
        self.has_header = kwargs.get('has_header', True)
        self.separator = kwargs.get('separator', ',')
        self.encoding = kwargs.get('encoding', 'utf-8')
        
        print("🔗 Graph CSV Loader")
        print("========================================")
        
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load CSV data and convert to graph format.
        
        Returns:
            Tuple of (graph_data, labels)
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
        
        # Convert to numpy arrays
        X = features.values.astype(np.float32)
        y = target.values
        
        # Normalize features for better similarity computation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create graph structure
        print(f"🔗 Creating graph structure...")
        edge_index = self._create_edges(X_scaled)
        
        # Convert to PyTorch tensors
        node_features = torch.tensor(X, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        labels = torch.tensor(y, dtype=torch.long)
        
        # Create PyG Data object
        graph_data = Data(x=node_features, edge_index=edge_index_tensor, y=labels)
        
        print(f"✅ Graph created:")
        print(f"   - Nodes: {graph_data.num_nodes}")
        print(f"   - Edges: {graph_data.num_edges}")
        print(f"   - Node features: {graph_data.num_node_features}")
        print(f"   - Classes: {len(np.unique(y))}")
        
        return graph_data, labels
        
    def _create_edges(self, features: np.ndarray) -> List[Tuple[int, int]]:
        """
        Create edges between nodes based on feature similarity.
        
        Args:
            features: Node feature matrix (normalized)
            
        Returns:
            List of edge tuples (source, target)
        """
        n_nodes = len(features)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(features)
        
        edges = []
        for i in range(n_nodes):
            # Get indices of most similar nodes
            similarities = similarity_matrix[i]
            # Exclude self-similarity
            similarities[i] = -1
            
            # Get top-k most similar nodes above threshold
            similar_indices = np.argsort(similarities)[::-1]
            edge_count = 0
            
            for j in similar_indices:
                if edge_count >= self.max_edges_per_node:
                    break
                if similarities[j] > self.edge_threshold:
                    edges.append((i, j))
                    edge_count += 1
                    
        print(f"   - Created {len(edges)} edges with similarity threshold {self.edge_threshold}")
        return edges
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the graph dataset."""
        return {
            "loader_type": "graph_csv",
            "file_path": self.file_path,
            "target_column": self.target_column,
            "edge_threshold": self.edge_threshold,
            "max_edges_per_node": self.max_edges_per_node
        }
