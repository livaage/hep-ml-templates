from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from typing import Dict, Any, Optional
from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.decision_tree")
class DecisionTreeModel(ModelBlock):
    """
    Decision Tree Classifier using scikit-learn.
    
    A simple, interpretable model good for understanding feature importance
    and decision boundaries. Well-suited for both binary and multiclass classification.
    
    Example usage via config:
        model = DecisionTreeModel()
        model.build(config)
        model.fit(X, y)
        predictions = model.predict(X)
    """

    def __init__(self, **kwargs):
        """Initialize with optional parameters."""
        # Default parameters
        default_params = {
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'random_state': 42,
            'criterion': 'gini',
            'class_weight': None
        }
        
        # Merge with any provided kwargs
        self.params = {**default_params, **kwargs}
        self.model: DecisionTreeClassifier | None = None
        self._auto_built = False

    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build model with optional config override."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
        
        # Filter out non-sklearn parameters
        sklearn_params = {k: v for k, v in params.items() 
                         if k not in ['block', '_target_', 'name', 'description']}
            
        self.model = DecisionTreeClassifier(**sklearn_params)
        self._auto_built = True
        print(f"✅ Decision Tree model built with max_depth={params.get('max_depth')}, "
              f"criterion='{params.get('criterion')}', class_weight={params.get('class_weight')}")

    def fit(self, X, y) -> None:
        """Fit the model. Auto-builds with defaults if not already built."""
        if self.model is None:
            self.build()  # Auto-build with defaults
            
        print(f"🌳 Training Decision Tree on {X.shape[0]} samples, {X.shape[1]} features...")
        
        # Convert to numpy arrays if needed  
        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_values, y_values)
        
        print(f"✅ Decision Tree training completed!")
        print(f"   - Tree depth: {self.model.get_depth()}")
        print(f"   - Number of leaves: {self.model.get_n_leaves()}")

    def predict(self, X):
        """Make predictions. Returns probabilities for binary classification."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        # Convert to numpy arrays if needed
        X_values = X.values if hasattr(X, 'values') else X
        
        # Prefer probabilities if available; fallback to decision_function/labels
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_values)[:, 1]
        return self.model.predict(X_values)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        # Convert to numpy arrays if needed
        X_values = X.values if hasattr(X, 'values') else X
        
        return self.model.predict_proba(X_values)
