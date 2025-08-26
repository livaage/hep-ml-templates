from typing import Any, Dict, Optional
from xgboost import XGBClassifier
from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.xgb_classifier")
class XGBClassifierBlock(ModelBlock):
    """XGBoost classifier that works with or without configuration.

    Can be used standalone:
        model = XGBClassifierBlock()
        model.fit(X, y)  # Uses sensible defaults
        predictions = model.predict(X)

    Or with custom config:
        model = XGBClassifierBlock()
        model.build({'n_estimators': 200, 'max_depth': 8})
        model.fit(X, y)
    """

    def __init__(self, **kwargs):
        """Initialize with optional parameters."""
        # Default parameters for HEP use cases
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,  # Use all cores
            'eval_metric': 'logloss'
        }

        # Merge with any provided kwargs
        self.params = {**default_params, **kwargs}
        self.model: XGBClassifier | None = None

        # Auto-build with defaults if no explicit build() call
        self._auto_built = False

    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build model with optional config override."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
        self.model = XGBClassifier(**params)
        self._auto_built = True

    def fit(self, X, y) -> None:
        """Fit the model. Auto-builds with defaults if not already built."""
        if self.model is None:
            self.build()  # Auto-build with defaults
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions. Returns probabilities for binary classification."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")

        # Prefer probabilities if available; fallback to decision_function/labels
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
