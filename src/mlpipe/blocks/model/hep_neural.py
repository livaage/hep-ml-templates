"""
Transformer and CNN models specifically designed for HEP use cases.

Common applications:
- Jet classification and tagging
- Particle sequence modeling
- Event classification from detector images
- Multi-particle interaction analysis
"""

from typing import Any, Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.transformer_hep")
class HEPTransformerBlock(ModelBlock):
    """
    Transformer model for particle sequence analysis.
    
    Ideal for:
    - Jet constituent analysis (particles within jets)
    - Event sequence classification
    - Variable-length particle collections
    - Attention-based particle interaction modeling
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'sequence_length': 50,  # Max particles per event/jet
            'input_dim': 4,  # 4-momentum features typically
            'd_model': 128,  # Model dimension
            'nhead': 8,  # Number of attention heads
            'num_layers': 4,  # Number of transformer layers
            'dim_feedforward': 256,
            'dropout': 0.1,
            'output_dim': 2,  # Binary classification
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'normalize_inputs': True,
            'use_positional_encoding': True
        }
        
        self.params = {**default_params, **kwargs}
        self.model = None
        self.scaler = StandardScaler() if self.params['normalize_inputs'] else None
        self.trainer = None
        
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build Transformer model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        self.model = HEPTransformer(
            input_dim=params['input_dim'],
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'],
            output_dim=params['output_dim'],
            sequence_length=params['sequence_length'],
            learning_rate=params['learning_rate'],
            use_positional_encoding=params['use_positional_encoding']
        )
        
        self.trainer = pl.Trainer(
            max_epochs=params['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_progress_bar=True
        )
        
        print(f"✅ HEP Transformer built with {params['d_model']} model dimension, {params['nhead']} heads")
        
    def fit(self, X, y) -> None:
        """Fit the Transformer model."""
        # Prepare sequence data
        X_sequences, y_tensor = self._prepare_sequence_data(X, y)
        
        dataset = TensorDataset(X_sequences, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True
        )
        
        if self.model is None:
            self.build()
            
        print(f"🔄 Training Transformer on {X_sequences.shape[0]} sequences...")
        self.trainer.fit(self.model, dataloader)
        print("✅ Transformer training completed!")
        
    def predict(self, X):
        """Make predictions with the Transformer."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_sequences, _ = self._prepare_sequence_data(X)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_sequences), self.params['batch_size']):
                batch = X_sequences[i:i + self.params['batch_size']]
                output = self.model(batch)
                pred_proba = F.softmax(output, dim=1)
                predictions.extend(pred_proba[:, 1].cpu().numpy())
                
        return np.array(predictions)
    
    def _prepare_sequence_data(self, X, y=None):
        """Convert tabular data to sequence format for Transformer."""
        # This is a simplified example - real implementation would depend on data structure
        # For jets: group features by particle, pad/truncate to fixed length
        # For events: group particles by event
        
        if self.scaler and self.params['normalize_inputs'] and y is not None:
            X_scaled = self.scaler.fit_transform(X)
        elif self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X
            
        # Simple example: reshape to sequence format
        # Real implementation would group by event/jet ID
        n_samples = X_scaled.shape[0]
        n_features = X_scaled.shape[1]
        
        # Reshape assuming features represent flattened sequences
        features_per_particle = self.params['input_dim']
        seq_length = min(n_features // features_per_particle, self.params['sequence_length'])
        
        X_reshaped = X_scaled[:, :seq_length * features_per_particle].reshape(
            n_samples, seq_length, features_per_particle
        )
        
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
        
        if y is not None:
            if hasattr(y, 'values'):
                y_values = y.values
            else:
                y_values = y
            y_tensor = torch.tensor(y_values, dtype=torch.long)
            return X_tensor, y_tensor
        
        return X_tensor, None


class HEPTransformer(pl.LightningModule):
    """Transformer model for HEP sequence data."""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward,
                 dropout, output_dim, sequence_length, learning_rate=0.001,
                 use_positional_encoding=True):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x, mask=None):
        """Forward pass through Transformer."""
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding."""
        return x + self.pe[:, :x.size(1)]


@register("model.cnn_hep")
class HEPCNNBlock(ModelBlock):
    """
    1D CNN for HEP data analysis.
    
    Ideal for:
    - Calorimeter image analysis
    - Detector signal processing
    - Local pattern recognition in HEP data
    - Time series analysis of detector signals
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'input_channels': 1,
            'conv_layers': [16, 32, 64],
            'kernel_sizes': [3, 3, 3],
            'pool_sizes': [2, 2, 2],
            'fc_layers': [128, 64],
            'output_dim': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'normalize_inputs': True
        }
        
        self.params = {**default_params, **kwargs}
        self.model = None
        self.scaler = StandardScaler() if self.params['normalize_inputs'] else None
        self.trainer = None
        
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build CNN model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        self.model = HEPCNN(
            input_channels=params['input_channels'],
            conv_layers=params['conv_layers'],
            kernel_sizes=params['kernel_sizes'],
            pool_sizes=params['pool_sizes'],
            fc_layers=params['fc_layers'],
            output_dim=params['output_dim'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            input_length=params.get('input_length', 100)  # Will be set during fit
        )
        
        self.trainer = pl.Trainer(
            max_epochs=params['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_progress_bar=True
        )
        
        print(f"✅ HEP CNN built with {len(params['conv_layers'])} conv layers")
        
    def fit(self, X, y) -> None:
        """Fit the CNN model."""
        # Prepare data
        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X
        
        # Reshape for CNN (batch, channels, length)
        X_cnn = X_scaled.reshape(-1, self.params['input_channels'], X_scaled.shape[1] // self.params['input_channels'])
        
        # Update input length
        input_length = X_cnn.shape[2]
        if self.model is None:
            self.params['input_length'] = input_length
            self.build()
        
        self.model.input_length = input_length
        self.model.build_layers()
        
        X_tensor = torch.tensor(X_cnn, dtype=torch.float32)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)
        
        print(f"🔄 Training CNN on {X_cnn.shape[0]} samples...")
        self.trainer.fit(self.model, dataloader)
        print("✅ CNN training completed!")
        
    def predict(self, X):
        """Make predictions with the CNN."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X
            
        X_cnn = X_scaled.reshape(-1, self.params['input_channels'], X_scaled.shape[1] // self.params['input_channels'])
        X_tensor = torch.tensor(X_cnn, dtype=torch.float32)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.params['batch_size']):
                batch = X_tensor[i:i + self.params['batch_size']]
                output = self.model(batch)
                pred_proba = F.softmax(output, dim=1)
                predictions.extend(pred_proba[:, 1].cpu().numpy())
                
        return np.array(predictions)


class HEPCNN(pl.LightningModule):
    """1D CNN for HEP data."""
    
    def __init__(self, input_channels, conv_layers, kernel_sizes, pool_sizes,
                 fc_layers, output_dim, dropout, learning_rate, input_length):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_channels = input_channels
        self.conv_layers = conv_layers
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.fc_layers = fc_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.input_length = input_length
        
        self.build_layers()
        
    def build_layers(self):
        """Build CNN layers."""
        # Convolutional layers
        conv_modules = []
        in_channels = self.input_channels
        current_length = self.input_length
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(self.conv_layers, self.kernel_sizes, self.pool_sizes)
        ):
            # Conv layer
            conv_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_modules.append(nn.ReLU())
            conv_modules.append(nn.MaxPool1d(pool_size))
            conv_modules.append(nn.Dropout(self.dropout))
            
            in_channels = out_channels
            current_length = current_length // pool_size
            
        self.conv_net = nn.Sequential(*conv_modules)
        
        # Calculate flattened size
        self.flattened_size = in_channels * current_length
        
        # Fully connected layers
        fc_modules = []
        fc_dims = [self.flattened_size] + self.fc_layers + [self.output_dim]
        
        for i in range(len(fc_dims) - 1):
            fc_modules.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))
            if i < len(fc_dims) - 2:  # No activation on output layer
                fc_modules.append(nn.ReLU())
                fc_modules.append(nn.Dropout(self.dropout))
        
        self.fc_net = nn.Sequential(*fc_modules)
        
    def forward(self, x):
        """Forward pass through CNN."""
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_net(x)
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
