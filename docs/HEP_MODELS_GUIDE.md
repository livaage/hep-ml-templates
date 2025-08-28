# HEP ML Models Guide

This guide covers the expanded model library in hep-ml-templates, specifically designed for High Energy Physics applications.

## Available Models

### Traditional ML Models

#### 1. **XGBoost** (`model.xgb_classifier`)
- **Best for**: Tabular data, feature engineering competitions, robust baseline
- **Advantages**: Excellent performance, handles missing values, built-in regularization
- **HEP use cases**: Event classification, jet tagging, trigger decisions
- **Config**: `configs/model/xgb_classifier.yaml`

#### 2. **Decision Tree** (`model.decision_tree`)  
- **Best for**: Interpretable models, feature importance analysis
- **Advantages**: Easy to interpret, no assumptions about data distribution
- **HEP use cases**: Cut-based analysis automation, background rejection
- **Config**: `configs/model/decision_tree.yaml`

#### 3. **Random Forest** (`model.random_forest`)
- **Best for**: Robust baseline, handling mixed data types
- **Advantages**: Built-in feature importance, robust to outliers, minimal tuning
- **HEP use cases**: Multi-class particle identification, systematic uncertainty studies
- **Config**: `configs/model/random_forest.yaml`

#### 4. **Support Vector Machine** (`model.svm`)
- **Best for**: High-dimensional data, clear separation margins
- **Advantages**: Effective in high dimensions, memory efficient
- **HEP use cases**: High-dimensional feature spaces, small datasets with many features
- **Config**: `configs/model/svm.yaml`

#### 5. **Multi-Layer Perceptron** (`model.mlp`)
- **Best for**: Non-linear patterns, moderate-sized datasets
- **Advantages**: Can learn complex non-linear relationships, flexible architecture
- **HEP use cases**: Feature interaction learning, non-linear decision boundaries
- **Config**: `configs/model/mlp.yaml`

#### 6. **AdaBoost** (`model.adaboost`)
- **Best for**: Combining weak learners, binary classification
- **Advantages**: Reduces bias and variance, adaptive to data
- **HEP use cases**: Weak signal extraction, combining simple cuts
- **Config**: `configs/model/adaboost.yaml`

### Deep Learning Models

#### 7. **Vanilla Autoencoder** (`model.ae_vanilla`)
- **Best for**: Anomaly detection, dimensionality reduction
- **Advantages**: Unsupervised learning, compression, reconstruction
- **HEP use cases**: New physics searches, background modeling, detector anomalies
- **Config**: `configs/model/ae_vanilla.yaml`

#### 8. **Variational Autoencoder** (`model.ae_variational`)
- **Best for**: Generative modeling, probabilistic anomaly detection
- **Advantages**: Probabilistic latent space, generative capabilities
- **HEP use cases**: Synthetic data generation, probabilistic anomaly detection
- **Config**: `configs/model/ae_variational.yaml`

#### 9. **Graph Convolutional Network** (`model.gnn_gcn`)
- **Best for**: Particle interaction networks, detector geometries
- **Advantages**: Handles graph-structured data, permutation invariant
- **HEP use cases**: Jet constituent analysis, particle interaction modeling
- **Config**: `configs/model/gnn_gcn.yaml`

#### 10. **Graph Attention Network** (`model.gnn_gat`)
- **Best for**: Complex particle interactions with attention
- **Advantages**: Attention mechanism, interpretable interactions
- **HEP use cases**: Complex jet substructure, multi-particle correlations
- **Config**: `configs/model/gnn_gat.yaml`

#### 11. **Transformer** (`model.transformer_hep`)
- **Best for**: Particle sequences, variable-length collections
- **Advantages**: Self-attention, handles sequences, parallelizable
- **HEP use cases**: Jet constituent sequences, event-level analysis
- **Config**: `configs/model/transformer_hep.yaml`

#### 12. **1D CNN** (`model.cnn_hep`)
- **Best for**: Detector images, signal processing, local patterns
- **Advantages**: Translation invariant, local feature detection
- **HEP use cases**: Calorimeter analysis, detector signal processing
- **Config**: `configs/model/cnn_hep.yaml`

### Ensemble Methods

#### 13. **Voting Ensemble** (`model.ensemble_voting`)
- **Best for**: Combining multiple algorithms for robustness
- **Advantages**: Reduces overfitting, combines algorithm strengths
- **HEP use cases**: Final analysis combinations, systematic uncertainty reduction
- **Config**: `configs/model/ensemble_voting.yaml`

## Usage Examples

### Quick Start - Traditional Models
```bash
# Random Forest
mlpipe run --overrides model=random_forest

# SVM
mlpipe run --overrides model=svm

# Ensemble
mlpipe run --overrides model=ensemble_voting
```

### Deep Learning Models (requires torch installation)
```bash
# Install PyTorch dependencies
pip install -e '.[torch]'

# Autoencoder for anomaly detection
mlpipe run --overrides model=ae_vanilla

# GNN for graph data
pip install -e '.[gnn]'
mlpipe run --overrides model=gnn_gcn

# Transformer for sequences
mlpipe run --overrides model=transformer_hep
```

### Model Comparison Workflow
```bash
# Compare traditional methods
mlpipe run --overrides model=xgb_classifier
mlpipe run --overrides model=random_forest
mlpipe run --overrides model=svm

# Compare ensemble vs individual
mlpipe run --overrides model=ensemble_voting
```

## HEP-Specific Considerations

### For Jet Physics
- **Graph data**: Use `model.gnn_gcn` or `model.gnn_gat`
- **Sequences**: Use `model.transformer_hep` 
- **Traditional**: Use `model.xgb_classifier` or `model.random_forest`

### For Anomaly Detection
- **Primary**: Use `model.ae_vanilla` or `model.ae_variational`
- **Baseline**: Use `model.random_forest` with outlier scoring

### For High-Dimensional Data
- **Linear separable**: Use `model.svm` 
- **Non-linear**: Use `model.mlp` or neural networks

### For Interpretability
- **Primary**: Use `model.decision_tree` or `model.random_forest`
- **Feature importance**: Random Forest provides built-in importance scores

## Installation Options

```bash
# Traditional models only
pip install -e '.[ensemble]'

# Neural networks
pip install -e '.[torch]'

# Graph neural networks  
pip install -e '.[gnn]'

# Everything
pip install -e '.[all]'
```

## Performance Guidelines

### Model Selection by Dataset Size
- **Small (< 1K samples)**: SVM, Decision Tree
- **Medium (1K-100K)**: XGBoost, Random Forest, MLP
- **Large (> 100K)**: XGBoost, Neural Networks, Ensemble

### Model Selection by Data Type
- **Tabular**: XGBoost, Random Forest, SVM
- **Sequences**: Transformer, CNN
- **Graphs**: GNN (GCN/GAT)
- **Images**: CNN
- **Mixed**: Ensemble methods

### Computational Considerations
- **Fastest**: Decision Tree, Random Forest
- **Medium**: XGBoost, SVM, MLP
- **Slowest**: Deep learning models (but more flexible)

## Advanced Usage

### Custom Model Configuration
```yaml
# configs/model/my_custom_rf.yaml
block: model.random_forest
params:
  n_estimators: 500
  max_depth: 15
  min_samples_split: 10
  class_weight: "balanced"  # For imbalanced HEP data
```

### Hyperparameter Tuning
Each model supports extensive hyperparameter customization through YAML configs. See individual config files for all available parameters.

### Feature Engineering Integration
All models work seamlessly with the feature engineering pipeline:
```bash
mlpipe run --overrides model=random_forest feature_eng=my_features data=my_hep_data
```
