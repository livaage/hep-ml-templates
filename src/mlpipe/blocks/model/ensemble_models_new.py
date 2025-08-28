"""
Ensemble machine learning models for High Energy Physics data analysis.

This module contains ensemble methods that combine multiple base estimators
to improve generalizability and robustness.

Includes:
- Random Forest (bagging ensemble)
- AdaBoost (boosting ensemble)
- Voting Classifier (combines different algorithms)
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.random_forest")
class RandomForestBlock(ModelBlock):
    """
    Random Forest classifier - excellent baseline for HEP.
    
    Advantages:
    - Handles mixed data types well
    - Built-in feature importance
    - Robust to outliers
    - Good performance on tabular data
    - Minimal hyperparameter tuning needed
    
    Example usage:
        model = RandomForestBlock(n_estimators=100, max_depth=10)
        model.build()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': None,  # or 'balanced' for imbalanced datasets
            'criterion': 'gini'
        }
        
        self.params = {**default_params, **kwargs}
        self.model = None
        
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build Random Forest model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        # Filter sklearn parameters
        sklearn_params = {k: v for k, v in params.items() 
                         if k not in ['block', '_target_', 'name', 'description']}
        
        self.model = RandomForestClassifier(**sklearn_params)
        
        print(f"✅ Random Forest built with {params['n_estimators']} trees, "
              f"max_depth={params['max_depth']}")
        
    def fit(self, X, y) -> None:
        """Fit Random Forest model."""
        if self.model is None:
            self.build()
            
        print(f"🌲 Training Random Forest on {X.shape[0]} samples, {X.shape[1]} features...")
        
        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_values, y_values)
        
        # Print feature importances
        if hasattr(X, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("✅ Random Forest training completed!")
            print("📊 Top 5 most important features:")
            print(feature_importance.head().to_string(index=False))
        else:
            print("✅ Random Forest training completed!")
            
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_values)[:, 1]
        return self.model.predict(X_values)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_values)
    
    def get_feature_importance(self):
        """Get feature importances."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
        return self.model.feature_importances_


@register("model.adaboost")
class AdaBoostBlock(ModelBlock):
    """
    AdaBoost classifier - adaptive boosting ensemble.
    
    Good for:
    - Weak learner combination
    - Reducing bias and variance
    - Binary classification problems
    - When you have a good base classifier
    
    Example usage:
        model = AdaBoostBlock(n_estimators=50, learning_rate=1.0)
        model.build()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            # 'algorithm': 'SAMME.R',  # Deprecated in sklearn 1.6+
            'random_state': 42
        }
        
        self.params = {**default_params, **kwargs}
        self.model = None
        
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build AdaBoost model."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        sklearn_params = {k: v for k, v in params.items() 
                         if k not in ['block', '_target_', 'name', 'description']}
        
        self.model = AdaBoostClassifier(**sklearn_params)
        
        print(f"✅ AdaBoost built with {params['n_estimators']} estimators, "
              f"learning_rate={params['learning_rate']}")
        
    def fit(self, X, y) -> None:
        """Fit AdaBoost model."""
        if self.model is None:
            self.build()
            
        print(f"🚀 Training AdaBoost on {X.shape[0]} samples, {X.shape[1]} features...")
        
        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_values, y_values)
        
        print("✅ AdaBoost training completed!")
        print(f"   - Estimator weights shape: {self.model.estimator_weights_.shape}")
        
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_values)[:, 1]
        return self.model.predict(X_values)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_values)


@register("model.ensemble_voting")
class VotingEnsembleBlock(ModelBlock):
    """
    Voting ensemble combining multiple models.
    
    Combines predictions from different algorithms to improve robustness.
    Available models include XGBoost, Random Forest, SVM, and MLP.
    
    Good for:
    - Improving robustness
    - Combining different algorithm strengths
    - Reducing overfitting
    
    Example usage:
        model = VotingEnsembleBlock(voting='soft', use_xgb=True, use_rf=True)
        model.build()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """
    
    def __init__(self, **kwargs):
        default_params = {
            'voting': 'soft',  # 'hard' or 'soft'
            'weights': None,  # Equal weights by default
            'use_xgb': True,
            'use_rf': True,
            'use_svm': False,  # Requires SVM module
            'use_mlp': False,  # Requires MLP module
            'random_state': 42
        }
        
        self.params = {**default_params, **kwargs}
        self.models = {}
        self.scaler = None
        self.fitted = False
        
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build ensemble of models."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        from sklearn.ensemble import VotingClassifier
        
        # Initialize base models
        estimators = []
        
        if params['use_xgb']:
            try:
                from xgboost import XGBClassifier
                xgb = XGBClassifier(random_state=params['random_state'], eval_metric='logloss')
                estimators.append(('xgb', xgb))
            except ImportError:
                print("Warning: XGBoost not available, skipping from ensemble")
        
        if params['use_rf']:
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=params['random_state'],
                n_jobs=-1
            )
            estimators.append(('rf', rf))
        
        if params['use_svm']:
            try:
                from mlpipe.blocks.model.svm import SVMBlock
                svm_block = SVMBlock(probability=True, random_state=params['random_state'])
                svm_block.build()
                estimators.append(('svm', svm_block.model))
                # Initialize scaler for SVM
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            except ImportError:
                print("Warning: SVM model not available, skipping from ensemble")
        
        if params['use_mlp']:
            try:
                from mlpipe.blocks.model.mlp import MLPBlock
                mlp_block = MLPBlock(random_state=params['random_state'])
                mlp_block.build()
                estimators.append(('mlp', mlp_block.model))
                # Initialize scaler for MLP
                if self.scaler is None:
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
            except ImportError:
                print("Warning: MLP model not available, skipping from ensemble")
        
        if not estimators:
            raise ValueError("No base models available for ensemble")
            
        self.model = VotingClassifier(
            estimators=estimators,
            voting=params['voting'],
            weights=params['weights']
        )
        
        print(f"✅ Voting Ensemble built with {len(estimators)} models: "
              f"{[name for name, _ in estimators]}")
        
    def fit(self, X, y) -> None:
        """Fit ensemble model."""
        if self.model is None:
            self.build()
            
        print(f"🗳️  Training Ensemble on {X.shape[0]} samples, {X.shape[1]} features...")
        
        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y
        
        # Scale features if needed (for SVM/MLP components)
        if self.scaler is not None:
            X_processed = self.scaler.fit_transform(X_values)
        else:
            X_processed = X_values
        
        self.model.fit(X_processed, y_values)
        self.fitted = True
        
        print("✅ Ensemble training completed!")
        
    def predict(self, X):
        """Make ensemble predictions."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        
        # Scale features if needed
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_values)
        else:
            X_processed = X_values
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_processed)[:, 1]
        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        
        # Scale features if needed
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_values)
        else:
            X_processed = X_values
        
        return self.model.predict_proba(X_processed)
