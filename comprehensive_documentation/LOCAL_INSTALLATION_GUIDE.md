# HEP-ML-Templates Local Installation Guide

## Overview

This guide demonstrates how to integrate HEP-ML-Templates model blocks into your existing machine learning pipeline with minimal code changes. The library provides modular ML blocks that can be easily adopted in any HEP analysis workflow.

## Prerequisites

- Python 3.8+
- Your existing ML pipeline/project
- Basic understanding of Python and machine learning workflows

## Installation Options

### Option 1: Local Development Installation (Recommended for Testing)

If you have the hep-ml-templates source code locally:

```bash
# Navigate to your project directory
cd /path/to/your/project

# Install specific model extras locally
mlpipe install-local --target-dir . <extra-name>

# Install the local package
pip install -e .
```

### Option 2: PyPI Installation (When Published)

```bash
# Install the base library (future PyPI release)
pip install hep-ml-templates

# Use the CLI to install specific extras
mlpipe install-local --target-dir . <extra-name>
pip install -e .
```

## Available Model Extras

Check available models:
```bash
mlpipe list-extras
```

### Individual Models
- `decision-tree` - Decision Tree classifier with preprocessing
- `random-forest` - Random Forest ensemble model  
- `xgb` - XGBoost gradient boosting
- `svm` - Support Vector Machine
- `mlp` - Multi-Layer Perceptron neural network
- `gnn` - Graph Neural Network for HEP data
- `ensemble` - Voting ensemble classifier

### Complete Pipelines
- `pipeline-decision-tree` - Full pipeline with decision tree
- `pipeline-xgb` - Complete XGBoost pipeline
- `pipeline-gnn` - Graph neural network pipeline

## Integration Workflow

### Step 1: Explore Available Models

```bash
# List all available extras
mlpipe list-extras

# Get details about a specific model
mlpipe extra-details random-forest

# Preview what will be installed
mlpipe preview-install random-forest
```

### Step 2: Install Your Chosen Model

```bash
# Install the model locally
mlpipe install-local --target-dir . random-forest

# Install as editable package
pip install -e .
```

### Step 3: Minimal Code Integration

Replace your existing model with a few lines:

**Before (Traditional scikit-learn):**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Traditional approach
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict_proba(X_test_scaled)[:, 1]
```

**After (HEP-ML-Templates):**
```python
from mlpipe.blocks.model.ensemble_models import RandomForestBlock
from mlpipe.blocks.preprocessing.standard_scaler import StandardScaler

# HEP-ML-Templates approach
config = {'n_estimators': 100, 'random_state': 42}
model = RandomForestBlock()
model.build(config)
model.fit(X_train, y_train)  # Handles preprocessing internally
predictions = model.predict_proba(X_test)[:, 1]
```

### Step 4: Validate Installation

```bash
# Validate that all extras are properly installed
mlpipe validate-extras
```

## Minimal Integration Examples

### Example 1: Existing Training Script

If you have an existing training script, you typically only need to change 2-3 lines:

```python
# Existing imports (keep these)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Add hep-ml-templates import
from mlpipe.blocks.model.ensemble_models import RandomForestBlock

# Your existing data loading code (no changes needed)
X_train = pd.read_csv('train_features.csv')
y_train = pd.read_csv('train_labels.csv')

# Replace your model initialization (1-2 lines changed)
model = RandomForestBlock()
model.build({'n_estimators': 100, 'max_depth': 10})

# Your existing training and evaluation code (no changes needed)
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, predictions)
print(f"AUC: {auc:.4f}")
```

### Example 2: Jupyter Notebook Integration

```python
# Cell 1: Install extras (run once)
!mlpipe install-local --target-dir . mlp
!pip install -e .

# Cell 2: Import and use (minimal changes to existing code)
from mlpipe.blocks.model.mlp import MLPBlock

# Your existing data preparation stays the same
# ...

# Just change the model initialization
model = MLPBlock()
model.build({
    'hidden_sizes': [128, 64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001
})

# Everything else stays the same
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Testing Different Models

The beauty of HEP-ML-Templates is that you can easily test different models by changing just the import and model initialization:

```python
# Test 1: Decision Tree
from mlpipe.blocks.model.decision_tree import DecisionTreeModel
model = DecisionTreeModel()
model.build({'max_depth': 10})

# Test 2: Random Forest (just change 2 lines)
from mlpipe.blocks.model.ensemble_models import RandomForestBlock
model = RandomForestBlock()
model.build({'n_estimators': 100})

# Test 3: XGBoost (just change 2 lines)
from mlpipe.blocks.model.xgb_classifier import XGBClassifierBlock
model = XGBClassifierBlock()
model.build({'n_estimators': 100, 'learning_rate': 0.1})

# The rest of your code stays identical
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)[:, 1]
```

## Advanced Usage

### Using Complete Pipelines

```bash
# Install a complete pipeline with preprocessing
mlpipe install-local --target-dir . pipeline-xgb
pip install -e .
```

```python
# Use the full pipeline
from mlpipe.blocks.model.xgb_classifier import XGBClassifierBlock
from mlpipe.blocks.preprocessing.standard_scaler import StandardScaler

# Preprocessing + Model in one workflow
scaler = StandardScaler()
scaler.build({})
X_train_scaled = scaler.fit_transform(X_train)

model = XGBClassifierBlock() 
model.build({'n_estimators': 200})
model.fit(X_train_scaled, y_train)
```

### Configuration Management

All models accept flexible configuration:

```python
# Simple configuration
model.build({'n_estimators': 100})

# Advanced configuration
config = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'random_state': 42,
    'n_jobs': -1
}
model.build(config)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you installed the extra and ran `pip install -e .`
   ```bash
   mlpipe validate-extras
   ```

2. **Model Not Found**: Check available models
   ```bash
   mlpipe list-extras
   ```

3. **Configuration Error**: Check model details
   ```bash
   mlpipe extra-details <model-name>
   ```

### Verification Commands

```bash
# Check installation status
mlpipe validate-extras

# List installed extras
ls -la mlpipe/blocks/model/

# Test import
python -c "from mlpipe.blocks.model.ensemble_models import RandomForestBlock; print('✅ Import successful')"
```

## Best Practices

1. **Start Simple**: Begin with basic models like decision-tree or random-forest
2. **Minimal Changes**: Only modify model initialization, keep existing data preprocessing
3. **Test Incrementally**: Install and test one model at a time
4. **Use Configuration**: Leverage the flexible config system for hyperparameter tuning
5. **Validate Everything**: Always run `mlpipe validate-extras` after installation

## Summary

HEP-ML-Templates enables easy adoption of state-of-the-art ML models in your existing workflow with minimal code changes:

1. **Install**: `mlpipe install-local --target-dir . <model>`
2. **Import**: Change 1 import line
3. **Initialize**: Change 1-2 model setup lines  
4. **Run**: Everything else stays the same

This approach allows researchers to experiment with different models quickly while maintaining their existing analysis pipeline.
