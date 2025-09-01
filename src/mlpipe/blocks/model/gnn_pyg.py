"""Graph Neural Network implementations using PyTorch Geometric.

Common HEP use cases:
- Particle interaction networks
- Detector geometry analysis
- Jet classification and tagging
- Event topology classification
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.gnn_gcn")
class GCNClassifier(ModelBlock):
    """Graph Convolutional Network for node/graph classification.

    Ideal for:
    - Particle interaction networks
    - Jet constituent classification
    - Event-level classification from particle graphs
    """

    def __init__(self, **kwargs):
        default_params = {
            "input_dim": 4,  # Common for 4-momentum features
            "hidden_dims": [64, 32],
            "output_dim": 2,  # Binary classification
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "task": "node",  # 'node' or 'graph' - changed to 'node' for CSV data
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        self.params = {**default_params, **kwargs}
        self.model = None
        self.device = torch.device(self.params["device"])

    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build GCN model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params

        self.model = GCNNet(
            input_dim=params["input_dim"],
            hidden_dims=params["hidden_dims"],
            output_dim=params["output_dim"],
            dropout=params["dropout"],
            task=params["task"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        self.criterion = nn.CrossEntropyLoss()

        print(
            f"✅ GCN model built with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def fit(self, X, y) -> None:
        """Fit the GCN model."""
        if self.model is None:
            # Auto-detect input dimension from data
            if hasattr(X, "shape") and len(X.shape) > 1:
                self.params["input_dim"] = X.shape[1]
            elif isinstance(X, list) and len(X) > 0:
                self.params["input_dim"] = len(X[0]) if hasattr(X[0], "__len__") else X.shape[1]

            print(f"🔧 Auto-detected input dimension: {self.params['input_dim']}")
            self.build()

        # Convert data to PyG format
        data_list = self._prepare_graph_data(X, y)

        self.model.train()
        for epoch in range(self.params["epochs"]):
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
        """Convert tabular data to graph format."""
        # For CSV data: each row becomes a node, columns are node features
        # Create edges based on feature similarity or use a k-NN approach

        import numpy as np
        import pandas as pd
        from sklearn.neighbors import kneighbors_graph
        from sklearn.preprocessing import StandardScaler

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
        else:
            X_array = np.array(X, dtype=np.float32)

        # Standardize features for better similarity computation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)

        # Create edges using k-nearest neighbors (k=8 for good connectivity)
        k = min(8, len(X_array) - 1)  # Ensure k < num_samples
        adjacency = kneighbors_graph(
            X_scaled, n_neighbors=k, mode="connectivity", include_self=False
        )

        # Convert sparse adjacency matrix to edge list
        edge_indices = np.array(adjacency.nonzero()).T
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)

        # Node features are the original features
        node_features = torch.tensor(X_array, dtype=torch.float)

        # Create single graph with all nodes
        if y is not None:
            y_tensor = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
            data = Data(x=node_features, edge_index=edge_index, y=y_tensor)
        else:
            data = Data(x=node_features, edge_index=edge_index)

        print(
            f"🔗 Created graph: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_node_features} features per node"
        )

        return [data]  # Return single graph as list for consistency

    def _create_batches(self, data_list):
        """Create batches from graph data."""
        batch_size = self.params["batch_size"]
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i : i + batch_size]
            yield Batch.from_data_list(batch_data)


class GCNNet(nn.Module):
    """Graph Convolutional Network architecture."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2, task="graph"):
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
        if self.task == "graph":
            x = global_mean_pool(x, batch.batch)

        # Final classification
        x = self.classifier(x)
        return x


@register("model.gnn_gat")
class GATClassifier(GCNClassifier):
    """Graph Attention Network for more sophisticated attention-based learning.

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
            input_dim=params["input_dim"],
            hidden_dims=params["hidden_dims"],
            output_dim=params["output_dim"],
            dropout=params["dropout"],
            task=params["task"],
            heads=params.get("attention_heads", 4),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        self.criterion = nn.CrossEntropyLoss()

        print(
            f"✅ GAT model built with {sum(p.numel() for p in self.model.parameters())} parameters"
        )


class GATNet(nn.Module):
    """Graph Attention Network architecture."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2, task="graph", heads=4):
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
        if self.task == "graph":
            x = global_mean_pool(x, batch.batch)

        x = self.classifier(x)
        return x
