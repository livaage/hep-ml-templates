"""
Support Vector Machine model implementation for High Energy Physics data analysis.

This module provides a scikit-learn based SVM classifier optimized for HEP use cases.
"""

from typing import Any, Dict, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.svm")
class SVMBlock(ModelBlock):
    """
    Support Vector Machine - excellent for high-dimensional data.

    Good for:
    - High-dimensional feature spaces
    - When you have more features than samples
    - Clear margin of separation
    - Non-linear decision boundaries (with RBF kernel)

    Example usage:
        model = SVMBlock(C=1.0, kernel='rbf')
        model.build()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, **kwargs):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',  # 'linear', 'poly', 'rbf', 'sigmoid'
            'degree': 3,  # For poly kernel
            'gamma': 'scale',  # 'scale', 'auto' or float
            'probability': True,  # Enable probability estimates
            'random_state': 42,
            'class_weight': None,
            'cache_size': 200
        }

        self.params = {**default_params, **kwargs}
        self.model = None
        self.scaler = StandardScaler()  # SVM needs scaled features

    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build SVM model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params

        sklearn_params = {k: v for k, v in params.items()
                         if k not in ['block', '_target_', 'name', 'description']}

        self.model = SVC(**sklearn_params)

        print(f"✅ SVM built with {params['kernel']} kernel, C={params['C']}")

    def fit(self, X, y) -> None:
        """Fit SVM model with feature scaling."""
        if self.model is None:
            self.build()

        print(f"🔍 Training SVM on {X.shape[0]} samples, {X.shape[1]} features...")

        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y

        # Scale features (important for SVM)
        X_scaled = self.scaler.fit_transform(X_values)

        self.model.fit(X_scaled, y_values)

        print("✅ SVM training completed!")
        print(f"   - Support vectors: {self.model.n_support_}")

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
