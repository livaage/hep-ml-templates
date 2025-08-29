"""
Autoencoder implementations using PyTorch Lightning.

Common HEP use cases:
- Anomaly detection in particle physics
- Dimensionality reduction for high-dimensional detector data
- Background rejection and signal enhancement
- Unsupervised feature learning from raw detector data
"""

from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.ae_vanilla")
class VanillaAutoencoderBlock(ModelBlock):
    """
    Standard Autoencoder for dimensionality reduction and anomaly detection.

    Ideal for:
    - Anomaly detection in HEP data
    - Background suppression
    - Feature extraction from high-dimensional detector data
    """

    def __init__(self, **kwargs):
        default_params = {
            'encoder_layers': [128, 64, 32],
            'latent_dim': 16,
            'decoder_layers': [32, 64, 128],  # Mirror of encoder
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_epochs': 100,
            'dropout': 0.2,
            'weight_decay': 1e-5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'reconstruction_loss': 'mse',  # 'mse' or 'bce'
            'normalize_inputs': True
        }

        self.params = {**default_params, **kwargs}
        self.model = None
        self.scaler = StandardScaler() if self.params['normalize_inputs'] else None
        self.trainer = None

    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build autoencoder model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params

        self.model = VanillaAutoencoder(
            input_dim=params.get('input_dim', 20),  # Will be set during fit
            encoder_layers=params['encoder_layers'],
            latent_dim=params['latent_dim'],
            decoder_layers=params['decoder_layers'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            reconstruction_loss=params['reconstruction_loss']
        )

        # Setup Lightning trainer
        self.trainer = pl.Trainer(
            max_epochs=params['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True
        )

        print(f"✅ Vanilla Autoencoder built with latent dimension {params['latent_dim']}")

    def fit(self, X, y=None) -> None:
        """Fit the autoencoder (unsupervised learning)."""
        # Prepare data
        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X

        # Update input dimension if not set
        input_dim = X_scaled.shape[1]
        if self.model is None:
            self.params['input_dim'] = input_dim
            self.build()

        # Update model input dimension
        self.model.input_dim = input_dim
        self.model.build_layers()

        # Create data loader
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, X_tensor)  # Target = input for autoencoder
        dataloader = DataLoader(
            dataset,
            batch_size=self.params['batch_size'],
            shuffle=True
        )

        print(f"🔄 Training Autoencoder on {X_scaled.shape[0]} samples, {input_dim} features...")

        # Train model
        self.trainer.fit(self.model, dataloader)

        print("✅ Autoencoder training completed!")

    def predict(self, X):
        """Return reconstruction error for anomaly detection."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X) first.")

        # Prepare data
        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            # Return reconstruction error as anomaly score
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return mse.numpy()

    def encode(self, X):
        """Get latent representations."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X) first.")

        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_tensor)
            return latent.numpy()

    def reconstruct(self, X):
        """Get reconstructed inputs."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X) first.")

        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            if self.scaler and self.params['normalize_inputs']:
                reconstructed = self.scaler.inverse_transform(reconstructed.numpy())
            return reconstructed.numpy()


class VanillaAutoencoder(pl.LightningModule):
    """PyTorch Lightning Autoencoder module."""

    def __init__(self, input_dim, encoder_layers, latent_dim, decoder_layers,
                 dropout=0.2, learning_rate=0.001, weight_decay=1e-5,
                 reconstruction_loss='mse'):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.encoder_layers = encoder_layers
        self.latent_dim = latent_dim
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reconstruction_loss = reconstruction_loss

        self.build_layers()

    def build_layers(self):
        """Build encoder and decoder layers."""
        # Encoder
        encoder_dims = [self.input_dim] + self.encoder_layers + [self.latent_dim]
        encoder_layers = []

        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:  # No activation on latent layer
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(self.dropout))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_dims = [self.latent_dim] + self.decoder_layers + [self.input_dim]
        decoder_layers = []

        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:  # No activation on output layer
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(self.dropout))

        # Add appropriate output activation
        if self.reconstruction_loss == 'bce':
            decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through autoencoder."""
        z = self.encode(x)
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, _ = batch  # Target is same as input
        x_hat = self(x)

        if self.reconstruction_loss == 'mse':
            loss = F.mse_loss(x_hat, x)
        elif self.reconstruction_loss == 'bce':
            loss = F.binary_cross_entropy(x_hat, x)
        else:
            raise ValueError(f"Unknown loss: {self.reconstruction_loss}")

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


@register("model.ae_variational")
class VariationalAutoencoderBlock(ModelBlock):
    """
    Variational Autoencoder for generative modeling and anomaly detection.

    Better for:
    - Generating synthetic HEP data
    - More robust anomaly detection via probabilistic modeling
    - Understanding data distribution in latent space
    """

    def __init__(self, **kwargs):
        default_params = {
            'encoder_layers': [128, 64],
            'latent_dim': 16,
            'decoder_layers': [64, 128],
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_epochs': 100,
            'dropout': 0.2,
            'weight_decay': 1e-5,
            'beta': 1.0,  # KL divergence weight
            'normalize_inputs': True
        }

        self.params = {**default_params, **kwargs}
        self.model = None
        self.scaler = StandardScaler() if self.params['normalize_inputs'] else None
        self.trainer = None

    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build VAE model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params

        self.model = VariationalAutoencoder(
            input_dim=params.get('input_dim', 20),
            encoder_layers=params['encoder_layers'],
            latent_dim=params['latent_dim'],
            decoder_layers=params['decoder_layers'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            beta=params['beta']
        )

        self.trainer = pl.Trainer(
            max_epochs=params['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_progress_bar=True
        )

        print(f"✅ Variational Autoencoder built with latent dimension {params['latent_dim']}")

    def fit(self, X, y=None) -> None:
        """Fit the VAE."""
        # Prepare data
        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X

        input_dim = X_scaled.shape[1]
        if self.model is None:
            self.params['input_dim'] = input_dim
            self.build()

        self.model.input_dim = input_dim
        self.model.build_layers()

        # Create data loader
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

        print(f"🔄 Training Variational Autoencoder on {X_scaled.shape[0]} samples...")
        self.trainer.fit(self.model, dataloader)
        print("✅ VAE training completed!")

    def predict(self, X):
        """Return reconstruction error for anomaly detection."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X) first.")

        if self.scaler and self.params['normalize_inputs']:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return mse.numpy()


class VariationalAutoencoder(pl.LightningModule):
    """Variational Autoencoder with reparameterization trick."""

    def __init__(self, input_dim, encoder_layers, latent_dim, decoder_layers,
                 dropout=0.2, learning_rate=0.001, weight_decay=1e-5, beta=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.encoder_layers = encoder_layers
        self.latent_dim = latent_dim
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta

        self.build_layers()

    def build_layers(self):
        """Build encoder and decoder with reparameterization."""
        # Encoder (to mean and log variance)
        encoder_dims = [self.input_dim] + self.encoder_layers
        encoder_layers = []

        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(self.dropout))

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(encoder_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], self.latent_dim)

        # Decoder
        decoder_dims = [self.latent_dim] + self.decoder_layers + [self.input_dim]
        decoder_layers = []

        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(self.dropout))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        """Training step with ELBO loss."""
        x, _ = batch
        x_hat, mu, logvar = self(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss (ELBO)
        loss = recon_loss + self.beta * kl_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)

        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
