"""
Multi-layer Perceptron (MLP) model implementation for High Energy Physics data analysis.

This module provides a scikit-learn based MLP classifier optimized for HEP use cases.
"""

from typing import Any, Dict, Optional
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.mlp")
class MLPBlock(ModelBlock):
    """
    Multi-layer Perceptron - simple neural network.

    Good for:
    - Non-linear patterns
    - Moderate-sized datasets
    - When you need a simple neural network
    - Feature interaction learning

    Example usage:
        model = MLPBlock(hidden_layer_sizes=(100, 50), activation='relu')
        model.build()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, **kwargs):
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',  # 'identity', 'logistic', 'tanh', 'relu'
            'solver': 'adam',  # 'lbfgs', 'sgd', 'adam'
            'alpha': 0.0001,  # L2 regularization
            'learning_rate': 'constant',  # 'constant', 'invscaling', 'adaptive'
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10
        }

        self.params = {**default_params, **kwargs}
        self.model = None
        self.scaler = StandardScaler()  # Neural networks need scaled features

    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build MLP model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params

        sklearn_params = {k: v for k, v in params.items()
                         if k not in ['block', '_target_', 'name', 'description']}

        self.model = MLPClassifier(**sklearn_params)

        print(f"✅ MLP built with layers {params['hidden_layer_sizes']}, "
              f"activation={params['activation']}")

    def fit(self, X, y) -> None:
        """Fit MLP model with feature scaling."""
        if self.model is None:
            self.build()

        print(f"🧠 Training MLP on {X.shape[0]} samples, {X.shape[1]} features...")

        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y

        # Scale features
        X_scaled = self.scaler.fit_transform(X_values)

        self.model.fit(X_scaled, y_values)

        print("✅ MLP training completed!")
        print(f"   - Iterations: {self.model.n_iter_}")
        print(f"   - Final loss: {self.model.loss_:.6f}")

    def predict(self, X):
        """Make predictions with feature scaling."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")

        X_values = X.values if hasattr(X, 'values') else X
        X_scaled = self.scaler.transform(X_values)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)[:, 1]
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict class probabilities with feature scaling."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")

        X_values = X.values if hasattr(X, 'values') else X
        X_scaled = self.scaler.transform(X_values)

        return self.model.predict_proba(X_scaled)
