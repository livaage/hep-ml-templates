# Extended HEP Models - Quick Reference

## 🎯 Model Selection Guide

### By Use Case

**Tabular Event Data (Most Common)**
```bash
# Robust baseline - excellent for most HEP tasks
mlpipe run --overrides model=xgb_classifier

# Interpretable results with feature importance  
mlpipe run --overrides model=random_forest

# High-dimensional data with clear separation
mlpipe run --overrides model=svm
```

**Anomaly Detection (New Physics Searches)**
```bash
# Unsupervised anomaly detection
mlpipe run --overrides model=ae_vanilla

# Probabilistic anomaly detection with generation
mlpipe run --overrides model=ae_variational

# Tree-based anomaly scores
mlpipe run --overrides model=random_forest
```

**Graph/Network Data (Jets, Particle Interactions)**
```bash
# Particle interaction networks
mlpipe run --overrides model=gnn_gcn

# Complex interactions with attention
mlpipe run --overrides model=gnn_gat
```

**Sequential Data (Particle Sequences)**
```bash
# Jet constituent sequences
mlpipe run --overrides model=transformer_hep

# Detector signal processing
mlpipe run --overrides model=cnn_hep
```

**Maximum Performance (Ensemble)**
```bash
# Combines XGBoost + Random Forest + SVM
mlpipe run --overrides model=ensemble_voting
```

## 📊 Performance vs Complexity

| Model | Training Speed | Prediction Speed | Interpretability | Performance | Memory |
|-------|---------------|------------------|------------------|-------------|---------|
| Decision Tree | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⚡⚡⚡ |
| Random Forest | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ⚡⚡ |
| XGBoost | ⚡⚡ | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ⚡⚡ |
| SVM | ⚡ | ⚡⚡ | ⭐ | ⭐⭐⭐ | ⚡ |
| MLP | ⚡ | ⚡⚡ | ⭐ | ⭐⭐ | ⚡ |
| Autoencoder | ⚡ | ⚡⚡ | ⭐ | ⭐⭐⭐ | ⚡ |
| GNN | ⚡ | ⚡ | ⭐ | ⭐⭐⭐ | ⚡ |
| Transformer | ⚡ | ⚡ | ⭐ | ⭐⭐⭐ | ⚡ |
| Ensemble | ⚡ | ⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ⚡ |

## 🚀 Installation by Model Type

```bash
# Traditional ML (fastest to get started)
cd hep-ml-templates
pip install -e '.[ensemble]'

# Neural Networks (requires PyTorch)
pip install -e '.[torch]'

# Graph Neural Networks (requires PyTorch Geometric)  
pip install -e '.[gnn]'

# Everything (full installation)
pip install -e '.[all]'
```

## 🎛️ Quick Configuration Examples

### High Performance Setup
```yaml
# configs/model/my_xgb_optimized.yaml
block: model.xgb_classifier
params:
  n_estimators: 500
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
```

### Interpretable Setup
```yaml
# configs/model/my_interpretable_rf.yaml
block: model.random_forest
params:
  n_estimators: 100
  max_depth: 8
  min_samples_split: 10
  max_features: "sqrt"
```

### Anomaly Detection Setup
```yaml
# configs/model/my_anomaly_ae.yaml
block: model.ae_vanilla
params:
  encoder_layers: [64, 32, 16]
  latent_dim: 8
  max_epochs: 50
```

## 📈 Model Comparison Workflow

```bash
# Run systematic comparison
echo "=== XGBoost Baseline ===" 
mlpipe run --overrides model=xgb_classifier

echo "=== Random Forest Comparison ==="
mlpipe run --overrides model=random_forest

echo "=== SVM Comparison ==="
mlpipe run --overrides model=svm

echo "=== Ensemble (Best Performance) ==="
mlpipe run --overrides model=ensemble_voting
```

## 🔧 Advanced Usage

### Custom Ensemble
```yaml
# configs/model/my_custom_ensemble.yaml
block: model.ensemble_voting
params:
  voting: "soft"
  weights: [2, 1, 1]  # Prefer XGBoost
  use_xgb: true
  use_rf: true
  use_svm: false  # Skip SVM for speed
```

### Hyperparameter Sweeps
```bash
# Test different XGBoost depths
mlpipe run --overrides model=xgb_classifier model.params.max_depth=4
mlpipe run --overrides model=xgb_classifier model.params.max_depth=8
mlpipe run --overrides model=xgb_classifier model.params.max_depth=12
```

## 🧪 Model Validation

Test your setup:
```bash
cd hep-ml-templates
python test_new_models.py
```

This will validate:
- ✅ Model imports work correctly
- ✅ All models implement the required interface
- ✅ Dependencies are available
- ⚠️  Missing dependencies are identified

## 📚 Next Steps

1. **Start Simple**: Begin with `model=xgb_classifier` or `model=random_forest`
2. **Compare**: Try different models on your data
3. **Optimize**: Use ensemble methods for maximum performance
4. **Specialize**: Use neural networks for complex data types
5. **Deploy**: Export your best model as a standalone project

For detailed documentation, see: `docs/HEP_MODELS_GUIDE.md`
