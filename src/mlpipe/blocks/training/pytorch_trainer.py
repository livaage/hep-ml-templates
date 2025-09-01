"""
PyTorch Lightning trainer for deep learning models in HEP-ML-Templates.

This trainer handles:
- PyTorch Lightning models
- Autoencoder training
- Neural network training with proper GPU/CPU handling
- Integration with HEP ML workflows
"""

from typing import Dict, Any
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from mlpipe.core.interfaces import Trainer
from mlpipe.core.registry import register


@register("train.pytorch")
class PyTorchTrainer(Trainer):
    """
    PyTorch Lightning trainer for deep learning models.
    
    Handles training of PyTorch Lightning models including:
    - Autoencoders for anomaly detection
    - Neural networks for classification
    - Proper data loading and validation splits
    """

    def __init__(self, **kwargs):
        default_params = {
            'max_epochs': 100,
            'batch_size': 64,
            'validation_split': 0.2,
            'accelerator': 'auto',  # Let Lightning choose best available
            'devices': 1,
            'log_every_n_steps': 50,
            'enable_progress_bar': True,
            'enable_model_summary': True
        }
        default_params.update(kwargs)
        self.params = default_params

    def train(self, model, X, y, config: Dict[str, Any]) -> Any:
        """
        Train a PyTorch Lightning model.
        
        Args:
            model: PyTorch Lightning module
            X: Training features
            y: Training targets (optional for autoencoders)
            config: Training configuration
            
        Returns:
            Trained PyTorch Lightning model
        """
        print(f"🔥 Training {model.__class__.__name__} on {len(X)} samples, {X.shape[1]} features...")
        
        # Convert DataFrames/Series to numpy arrays first
        X_np = X.values if hasattr(X, 'values') else X
        
        # Extract PyTorch Lightning model from ModelBlock wrapper
        if hasattr(model, 'model') and model.model is not None:
            # If it's a ModelBlock with a Lightning model inside
            lightning_model = model.model
            
            # Ensure the model knows the correct input dimension
            input_dim = X_np.shape[1]
            if hasattr(lightning_model, 'input_dim') and lightning_model.input_dim != input_dim:
                print(f"⚠️  Updating model input dimension from {lightning_model.input_dim} to {input_dim}")
                lightning_model.input_dim = input_dim
                lightning_model.build_layers()
        else:
            # If it's already a Lightning model
            lightning_model = model
        
        # Convert numpy arrays to tensors
        X_tensor = torch.FloatTensor(X_np)
        
        # For autoencoders, y is typically None or same as X
        if y is not None:
            y_np = y.values if hasattr(y, 'values') else y
            if hasattr(model, 'task_type') and model.task_type == 'autoencoder':
                # For autoencoders, target is same as input
                y_tensor = X_tensor
            else:
                # For classification/regression
                y_tensor = torch.LongTensor(y_np) if y_np.dtype == 'int64' else torch.FloatTensor(y_np)
        else:
            # Autoencoder case - reconstruct input
            y_tensor = X_tensor

        # Create train/validation split
        if self.params['validation_split'] > 0:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_tensor, y_tensor, 
                test_size=self.params['validation_split'], 
                random_state=42,
                stratify=y_np if y is not None and hasattr(model, 'task_type') and model.task_type != 'autoencoder' else None
            )
        else:
            X_train_split, y_train_split = X_tensor, y_tensor
            X_val_split = y_val_split = None

        # Create data loaders
        train_dataset = TensorDataset(X_train_split, y_train_split)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True
        )

        val_loader = None
        if X_val_split is not None:
            val_dataset = TensorDataset(X_val_split, y_val_split)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.params['batch_size'], 
                shuffle=False
            )

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.params['max_epochs'],
            accelerator=self.params['accelerator'],
            devices=self.params['devices'],
            log_every_n_steps=self.params['log_every_n_steps'],
            enable_progress_bar=self.params['enable_progress_bar'],
            enable_model_summary=self.params['enable_model_summary']
        )

        # Train the model
        # Train the model using PyTorch Lightning Trainer
        trainer.fit(lightning_model, train_loader, val_loader)
        
        print(f"✅ {model.__class__.__name__} training completed!")
        if hasattr(model, 'task_type') and model.task_type == 'autoencoder':
            print(f"   - Final reconstruction loss: {trainer.callback_metrics.get('train_loss', 'N/A')}")
        else:
            print(f"   - Final loss: {trainer.callback_metrics.get('train_loss', 'N/A')}")
        
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return training configuration."""
        return {
            "block": "training.pytorch",
            "params": self.params
        }
