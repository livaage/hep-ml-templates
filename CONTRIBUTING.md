# Contributing to HEP-ML-Templates

## Adding New Models to the Library

This guide explains how contributors can add new machine learning models to the hep-ml-templates library, ensuring proper integration with the local installation system and maintaining the library's modularity.

## Overview

To add a new model to the library, you need to modify several files to ensure:
1. The model is properly registered and can be discovered
2. Users can install it locally via `mlpipe install-local`
3. The model follows the established patterns and interfaces
4. Dependencies are properly managed

## Step-by-Step Guide

### Step 1: Create the Model Implementation

Create a new Python file in `src/mlpipe/blocks/model/` for your model. Use the following template:

```python
"""
[Model Name] implementation for High Energy Physics data analysis.

This module provides a [brief description of the model].
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
# Import your model's dependencies here
from sklearn.ensemble import YourModelClass

from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register


@register("model.your_model_name")  # This is how users will reference your model
class YourModelBlock(ModelBlock):
    """
    Brief description of your model.
    
    Good for:
    - Use case 1
    - Use case 2
    - Use case 3
    
    Example usage:
        model = YourModelBlock(param1=value1, param2=value2)
        model.build()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """
    
    def __init__(self, **kwargs):
        """Initialize with default parameters."""
        default_params = {
            'param1': default_value1,
            'param2': default_value2,
            'random_state': 42,
            # Add all parameters your model supports
        }
        
        self.params = {**default_params, **kwargs}
        self.model = None
        # Add any additional attributes needed (e.g., scalers)
        
    def build(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Build the model with optional config override."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        # Filter out non-model parameters
        sklearn_params = {k: v for k, v in params.items() 
                         if k not in ['block', '_target_', 'name', 'description']}
        
        self.model = YourModelClass(**sklearn_params)
        
        print(f"✅ {self.__class__.__name__} built with [key parameters]")
        
    def fit(self, X, y) -> None:
        """Fit the model."""
        if self.model is None:
            self.build()  # Auto-build if not already built
            
        print(f"🔄 Training {self.__class__.__name__} on {X.shape[0]} samples, {X.shape[1]} features...")
        
        # Convert to numpy arrays if needed
        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y
        
        # Apply any preprocessing if needed
        # X_processed = self.preprocess(X_values)
        
        self.model.fit(X_values, y_values)
        
        print("✅ Training completed!")
        # Add any model-specific metrics/info
        
    def predict(self, X):
        """Make predictions. Returns probabilities for binary classification."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        # X_processed = self.preprocess(X_values)
        
        # Prefer probabilities if available
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_values)[:, 1]
        return self.model.predict(X_values)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit(X, y) first.")
            
        X_values = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_values)
```

### Step 2: Create Configuration File

Create a YAML configuration file in `configs/model/your_model_name.yaml`:

```yaml
# @package _global_

# Model configuration for Your Model Name
model:
  _target_: mlpipe.blocks.model.your_model_file.YourModelBlock
  # List all parameters with their default values
  param1: default_value1
  param2: default_value2
  random_state: 42

# Optional: Add documentation
_help: |
  Configuration for Your Model Name
  
  Parameters:
  - param1: Description of parameter 1
  - param2: Description of parameter 2
```

### Step 3: Update pyproject.toml

Add your model to the optional dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
# Add your model to individual components
model-your-model = ["required-dependency>=1.0"]

# Add to algorithm-specific extras
your-model = ["required-dependency>=1.0", "scikit-learn>=1.2"]

# Add to complete pipeline bundles if applicable
pipeline-your-model = [
  "omegaconf>=2.3", "numpy>=1.22", "pandas>=2.0", 
  "scikit-learn>=1.2", "required-dependency>=1.0"
]

# Update 'all' extra to include your dependencies
all = [
  "omegaconf>=2.3", "numpy>=1.22", "pandas>=2.0", "scikit-learn>=1.2", 
  "xgboost>=1.7", "torch>=2.2", "lightning>=2.2", "torch-geometric>=2.5",
  "required-dependency>=1.0"  # Add your dependency here
]
```

### Step 4: Update Local Installation Mappings

Add your model to `src/mlpipe/cli/local_install.py` in the `EXTRAS_TO_BLOCKS` dictionary:

```python
EXTRAS_TO_BLOCKS = {
    # ... existing mappings ...
    
    # Individual component mapping
    'model-your-model': {
        'blocks': ['model/your_model_file.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/your_model_name.yaml']
    },
    
    # Algorithm-specific extra (shorthand)
    'your-model': {
        'blocks': ['model/your_model_file.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/your_model_name.yaml', 'preprocessing/standard.yaml']
    },
    
    # If it's part of a complete pipeline
    'pipeline-your-model': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'feature_eng/column_selector.py',
            'model/your_model_file.py',
            'evaluation/classification_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],
        'configs': [
            'pipeline.yaml',
            'data/higgs_uci.yaml',
            'preprocessing/standard.yaml',
            'model/your_model_name.yaml',
            'evaluation/classification.yaml'
        ],
        'data': ['HIGGS_100k.csv']
    },
}
```

### Step 5: Update Model Module Imports

Add your model import to `src/mlpipe/blocks/model/__init__.py`:

```python
try:
    from . import your_model_file         # registers "model.your_model_name"
except ImportError:
    pass  # Dependencies not available
```

### Step 6: Write Tests (Recommended)

Create tests for your model in `tests/model/test_your_model.py`:

```python
import pytest
import numpy as np
import pandas as pd
from mlpipe.blocks.model.your_model_file import YourModelBlock

def test_your_model_init():
    model = YourModelBlock()
    assert model.model is None
    assert 'param1' in model.params

def test_your_model_build():
    model = YourModelBlock()
    model.build()
    assert model.model is not None

def test_your_model_fit_predict():
    # Create dummy data
    X = pd.DataFrame(np.random.randn(100, 5))
    y = np.random.randint(0, 2, 100)
    
    model = YourModelBlock()
    model.build()
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(0 <= p <= 1 for p in predictions)  # For probability outputs
```

### Step 7: Update Documentation

Add your model to the README.md file:

1. Add it to the features list
2. Add installation instructions
3. Add usage examples
4. Update the model comparison table if present

## Model Categories and File Organization

### Ensemble Methods
File: `src/mlpipe/blocks/model/ensemble_models.py`
- Random Forest
- AdaBoost  
- Voting Classifier

### Individual Algorithms
Each gets its own file:
- `src/mlpipe/blocks/model/svm.py` - Support Vector Machine
- `src/mlpipe/blocks/model/mlp.py` - Multi-Layer Perceptron
- `src/mlpipe/blocks/model/decision_tree.py` - Decision Tree
- `src/mlpipe/blocks/model/xgb_classifier.py` - XGBoost

### Neural Networks
- `src/mlpipe/blocks/model/ae_lightning.py` - Autoencoders
- `src/mlpipe/blocks/model/hep_neural.py` - HEP-specific neural networks

## Best Practices

### 1. Error Handling
```python
def predict(self, X):
    if self.model is None:
        raise ValueError("Model not fitted. Call fit(X, y) first.")
    # ... rest of implementation
```

### 2. Input Validation
```python
def fit(self, X, y):
    # Convert pandas to numpy if needed
    X_values = X.values if hasattr(X, 'values') else X
    y_values = y.values if hasattr(y, 'values') else y
    
    # Validate shapes
    assert len(X_values) == len(y_values), "X and y must have same number of samples"
```

### 3. Logging and User Feedback
```python
print(f"✅ Model built with key_param={params['key_param']}")
print(f"🔄 Training on {X.shape[0]} samples, {X.shape[1]} features...")
print("✅ Training completed!")
```

### 4. Registry Naming Convention
- Use descriptive, consistent naming: `"model.your_model_name"`
- Follow existing patterns: `"model.random_forest"`, `"model.svm"`, etc.
- Avoid spaces and special characters

### 5. Configuration Flexibility
```python
def build(self, config: Optional[Dict[str, Any]] = None) -> None:
    if config:
        params = {**self.params, **config}
    else:
        params = self.params
    # This allows runtime configuration override
```

## Testing Your Model

After implementing your model, test the complete workflow:

1. **Test import and registration:**
   ```python
   from mlpipe.core.registry import list_blocks
   print("model.your_model_name" in list_blocks())
   ```

2. **Test local installation:**
   ```bash
   mlpipe install-local --target-dir test_install your-model
   cd test_install && pip install -e .
   ```

3. **Test integration:**
   ```python
   from mlpipe.blocks.model.your_model_file import YourModelBlock
   model = YourModelBlock()
   # Test full workflow
   ```

## Common Pitfalls

1. **Forgetting to update local_install.py** - Users won't be able to install your model locally
2. **Missing dependency handling** - Use try/except blocks for optional imports
3. **Inconsistent interfaces** - Always inherit from ModelBlock and implement required methods
4. **Poor error messages** - Provide clear, actionable error messages
5. **Missing documentation** - Document parameters, usage, and examples

## Getting Help

- Check existing model implementations for patterns
- Look at the ModelBlock interface definition
- Test with the provided example datasets
- Ask questions in GitHub issues before submitting PRs

## Review Process

Before submitting a PR:

1. Ensure all tests pass
2. Test local installation workflow
3. Update all relevant configuration files
4. Add comprehensive docstrings
5. Follow the existing code style
6. Add usage examples to documentation

Your contribution will be reviewed for:
- Code quality and consistency
- Integration with existing infrastructure  
- Documentation completeness
- Test coverage
- Performance considerations
