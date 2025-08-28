"""
Graph Neural Network implementations using PyTorch Geometric.

Common HEP use cases:
- Particle interaction networks
- Detector geometry analysis  
- Jet classification and tagging
- Event topology classification
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.gnn_gcn")
class GCNClassifier(ModelBlock):
    """
    Graph Convolutional Network for node/graph classification.
    
    Ideal for:
    - Particle interaction networks
    - Jet constituent classification
    - Event-level classification from particle graphs
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'input_dim': 4,  # Common for 4-momentum features
            'hidden_dims': [64, 32],
            'output_dim': 2,  # Binary classification
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'task': 'graph',  # 'node' or 'graph'
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.params = {**default_params, **kwargs}
        self.model = None
        self.device = torch.device(self.params['device'])
        
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build GCN model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        self.model = GCNNet(
            input_dim=params['input_dim'],
            hidden_dims=params['hidden_dims'],
            output_dim=params['output_dim'],
            dropout=params['dropout'],
            task=params['task']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=params['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✅ GCN model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def fit(self, X, y) -> None:
        """Fit the GCN model."""
        if self.model is None:
            self.build()
            
        # Convert data to PyG format
        data_list = self._prepare_graph_data(X, y)
        
        self.model.train()
        for epoch in range(self.params['epochs']):
            total_loss = 0
            for batch in self._create_batches(data_list):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(data_list):.4f}")
                
        print("✅ GCN training completed!")
        
    def predict(self, X):
        """Make predictions with the GCN model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        self.model.eval()
        data_list = self._prepare_graph_data(X)
        
        predictions = []
        with torch.no_grad():
            for batch in self._create_batches(data_list):
                batch = batch.to(self.device)
                out = self.model(batch)
                pred_proba = F.softmax(out, dim=1)
                predictions.extend(pred_proba[:, 1].cpu().numpy())
                
        return np.array(predictions)
    
    def _prepare_graph_data(self, X, y=None):
        """Convert tabular data to graph format for HEP use cases."""
        # This is a simplified example - real implementation would depend on data structure
        # For jets: each row could be a jet, columns could be constituent features
        # For events: each row could be a particle, need to group by event
        
        data_list = []
        for idx, row in X.iterrows():
            # Example: create a simple graph from tabular features
            # In practice, you'd use domain knowledge to construct meaningful graphs
            
            # Simple example: fully connected graph of features
            num_nodes = len(row)
            edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
            
            # Add reverse edges for undirected graph
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
            node_features = torch.tensor(row.values, dtype=torch.float).view(-1, 1)
            
            if y is not None:
                label = torch.tensor(y.iloc[idx], dtype=torch.long)
                data = Data(x=node_features, edge_index=edge_index, y=label)
            else:
                data = Data(x=node_features, edge_index=edge_index)
                
            data_list.append(data)
            
        return data_list
    
    def _create_batches(self, data_list):
        """Create batches from graph data."""
        batch_size = self.params['batch_size']
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i + batch_size]
            yield Batch.from_data_list(batch_data)


class GCNNet(nn.Module):
    """Graph Convolutional Network architecture."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2, task='graph'):
        super().__init__()
        self.task = task
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(GCNConv(dims[i], dims[i + 1]))
            
        self.conv_layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        # Apply GCN layers
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level prediction (pool node features)
        if self.task == 'graph':
            x = global_mean_pool(x, batch.batch)
            
        # Final classification
        x = self.classifier(x)
        return x


@register("model.gnn_gat")
class GATClassifier(GCNClassifier):
    """
    Graph Attention Network for more sophisticated attention-based learning.
    
    Better for:
    - Complex particle interaction patterns
    - Variable-size inputs
    - When attention weights provide interpretability
    """
    
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build GAT model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        self.model = GATNet(
            input_dim=params['input_dim'],
            hidden_dims=params['hidden_dims'],
            output_dim=params['output_dim'],
            dropout=params['dropout'],
            task=params['task'],
            heads=params.get('attention_heads', 4)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=params['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✅ GAT model built with {sum(p.numel() for p in self.model.parameters())} parameters")


class GATNet(nn.Module):
    """Graph Attention Network architecture."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2, task='graph', heads=4):
        super().__init__()
        self.task = task
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            if i == 0:
                # First layer with multiple heads
                layers.append(GATConv(dims[i], dims[i + 1], heads=heads, dropout=dropout))
            else:
                # Subsequent layers (account for head concatenation)
                layers.append(GATConv(dims[i] * heads, dims[i + 1], heads=1, dropout=dropout))
                
        self.conv_layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        # Apply GAT layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i == 0:  # First layer with multiple heads
                x = F.elu(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level prediction
        if self.task == 'graph':
            x = global_mean_pool(x, batch.batch)
            
        x = self.classifier(x)
        return x
