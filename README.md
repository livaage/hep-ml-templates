# HEP-ML-Templates

A **modular, plug-and-play machine learning framework** designed specifically for **High Energy Physics (HEP)** research. Build, test, and deploy ML models with true modularity - swap datasets, models, and preprocessing components with minimal code changes and zero vendor lock-in.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status:** Production-ready with comprehensive validation, beginner-tested setup (100% success rate, <10s per model), and real-world integration case studies demonstrating 3-line dataset swaps.

---

## ✨ Key Features

- 🧩 **True Modularity**: Mix and match components - datasets, models, preprocessing - independently with consistent APIs
- 🎯 **HEP-Optimized**: Built specifically for particle physics data and workflows, including HIGGS benchmark integration
- ⚡ **Rapid Prototyping**: Swap models/datasets with single CLI commands; beginner-tested setup averaging under 10 seconds per model
- 🔧 **Selective Installation**: Install only the components you need via curated "extras" system with preview and validation
- 🚀 **Dual CLI Interface**: Embedded `mlpipe` commands + optional standalone `mlpipe-manager` for flexibility
- 📊 **Standalone Projects**: Export templates to create self-contained, editable ML projects with no repository dependency
- 🤖 **Multi-Algorithm Support**: Traditional ML (XGBoost, Decision Trees, SVM) + Neural Networks (PyTorch, GNNs, Autoencoders)
- 📈 **Advanced Data Splitting**: Train/val/test splits with stratification, time-series support, and reproducible seeding

---

## 🏗️ Core Architecture

HEP-ML-Templates is built around four fundamental concepts:

### 1. **Blocks** - Modular Components
Self-contained Python classes with consistent APIs that hide library-specific details:

```python
from mlpipe.core.registry import register
from mlpipe.core.interfaces import ModelBlock

@register("model.decision_tree")
class DecisionTreeModel(ModelBlock):
    def build(self, config): ...
    def fit(self, X, y): ...
    def predict(self, X): ...
```

### 2. **Registry** - Discovery System
Unified discovery mechanism allowing code and configs to refer to blocks by name:

```yaml
# configs/model/decision_tree.yaml
block: model.decision_tree
max_depth: 10
criterion: gini
random_state: 42
```

### 3. **Configuration-First** - Reproducible Experiments
YAML-driven workflows with CLI overrides keep code stable while you iterate:

```bash
# Swap components at runtime
mlpipe run --overrides model=xgb_classifier data=higgs_100k
mlpipe run --overrides model.params.max_depth=8 preprocessing=time_series_split
```

### 4. **Extras System** - Selective Installation
Curated package sets map to concrete file collections with discovery, validation, and preview:

```bash
mlpipe list-extras                    # Discover available components
mlpipe extra-details model-xgb        # Inspect what's included
mlpipe preview-install model-xgb evaluation  # Preview before installing
mlpipe install-local model-xgb evaluation --target-dir ./my-project
```

---

## 🚀 Quick Start (30 seconds)

```bash
# 1) Clone & install the core library
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates
pip install -e .

# 2) Discover available components
mlpipe list-extras

# 3) Create a project with XGBoost + evaluation + HIGGS data
mlpipe install-local model-xgb evaluation data-higgs --target-dir ./my-hep-project
cd ./my-hep-project && pip install -e .

# 4) Run the pipeline (components are configurable)
mlpipe run --overrides model=xgb_classifier data=higgs_100k
```

**Alternative manager-style interface:**
```bash
mlpipe-manager list
mlpipe-manager details model-xgb
mlpipe-manager install model-xgb ./my-project
```

---

## 📊 Available Components (Blocks)

> Use `mlpipe list-extras` and `mlpipe extra-details <name>` for exact identifiers and installation details.

### 🎯 **Complete Pipelines** (4)
End-to-end workflows with model + preprocessing + evaluation:
- `pipeline-xgb` - XGBoost pipeline with preprocessing and metrics
- `pipeline-decision-tree` - Decision tree complete workflow 
- `pipeline-torch` - PyTorch neural network pipeline
- `pipeline-gnn` - Graph neural network pipeline

### 🧠 **Individual Models** (11)
Single algorithms with unified interfaces:

**Traditional ML:**
- `model-decision-tree`, `model-random-forest`, `model-svm`
- `model-xgb`, `model-mlp`, `model-adaboost`, `model-ensemble`

**Neural & Graph Models:**
- `model-torch` (PyTorch neural networks)
- `model-cnn` (Convolutional networks)
- `model-gnn` (Graph neural networks via PyTorch Geometric)
- `model-transformer` (Transformer architectures)

### ⚡ **Algorithm Combos** (9)
Model + preprocessing bundles for quick setup:
- `xgb`, `decision-tree`, `random-forest`, `svm`, `mlp`
- `ensemble`, `torch`, `gnn`, `adaboost`

### 📊 **Data Sources** (3)
- `data-higgs` - HIGGS benchmark dataset (validated integration)
- `data-csv` - Universal CSV loader with flexible configuration
- `data-split` - Advanced train/val/test splitting utilities

### 🏗️ **Component Categories** (3)
- `preprocessing` - Scaling, feature engineering, data splitting
- `evaluation` - Classification metrics, reconstruction metrics
- `feature-eng` - Feature engineering demonstrations

### 🌟 **Special** (1)
- `all` - Complete bundle (16 blocks, 27 configurations)

---

## 🛠️ Three Core Workflows

### 1. **Rapid Prototyping**
Experiment with different models and datasets using config/CLI overrides:

```bash
# Try different models on the same data
mlpipe run --overrides model=decision_tree
mlpipe run --overrides model=xgb_classifier
mlpipe run --overrides model=random_forest

# Switch datasets and preprocessing
mlpipe run --overrides data=csv_demo preprocessing=time_series_split
mlpipe run --overrides data=higgs_100k feature_eng=demo_features
```

### 2. **Standalone Project Scaffolding** 
Create self-contained projects with selected components:

```bash
# Create a new project directory
mlpipe install-local model-random-forest data-higgs evaluation --target-dir ./research-project
cd ./research-project && pip install -e .

# Add more components later
mlpipe install-local model-xgb preprocessing .
mlpipe run --overrides model=xgb_classifier preprocessing=stratified_split
```

### 3. **Integration into Existing Code**
Drop in individual blocks with minimal changes (~3 lines):

**Before (traditional scikit-learn):**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict_proba(X_test_scaled)[:, 1]
```

**After (with hep-ml-templates):**
```python
from mlpipe.blocks.model.ensemble_models import RandomForestBlock  # Change 1

config = {'n_estimators': 100, 'random_state': 42}
model = RandomForestBlock()                                        # Change 2
model.build(config)
model.fit(X_train, y_train)                                        # Change 3 - preprocessing handled internally
predictions = model.predict_proba(X_test)[:, 1]
```

**Swap to XGBoost:**
```python
from mlpipe.blocks.model.xgb_classifier import XGBClassifierBlock  # Only import changes
model = XGBClassifierBlock()                                       # Only class name changes
model.build({'n_estimators': 200, 'learning_rate': 0.1})
```

---

## 🔄 Advanced Data Splitting

Built-in splitting utilities with comprehensive support:

### **Convenience Function:**
```python
from mlpipe.blocks.preprocessing.data_split import split_data

splits = split_data(X, y, 
    train_size=0.7, val_size=0.15, test_size=0.15, 
    stratify=True, random_state=42
)
X_train, y_train = splits['train']
X_val, y_val = splits['val']
X_test, y_test = splits['test']
```

### **Class-Based Approach:**
```python
from mlpipe.blocks.preprocessing.data_split import DataSplitter

splitter = DataSplitter({
    'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
    'stratify': True, 'time_series': False, 'random_state': 42
})
splits = splitter.fit_transform(X, y)
```

### **Pipeline Integration:**
```bash
# Use pre-configured splitting strategies
mlpipe run --overrides preprocessing=train_val_test_split
mlpipe run --overrides preprocessing=stratified_split
mlpipe run --overrides preprocessing=time_series_split
```

### **Configuration Examples:**

**Stratified 70/15/15 Split:**
```yaml
# configs/preprocessing/stratified_split.yaml
train_size: 0.7
val_size: 0.15
test_size: 0.15
stratify: true
shuffle: true
random_state: 42
```

**Time Series Split (No Shuffle):**
```yaml
# configs/preprocessing/time_series_split.yaml
train_size: 0.7
val_size: 0.15
test_size: 0.15
time_series: true
shuffle: false
time_column: "timestamp"  # optional
```

---

## 💻 CLI Reference

### **Embedded CLI (`mlpipe`)**
```bash
# Discovery & Planning
mlpipe list-extras                                      # Show all available extras
mlpipe extra-details model-xgb                         # Inspect specific extra
mlpipe preview-install model-xgb evaluation            # Preview installation
mlpipe validate-extras                                  # Validate extras system

# Installation & Setup
mlpipe install-local model-xgb data-higgs --target-dir ./project

# Execution & Experimentation
mlpipe run                                              # Use default pipeline
mlpipe run --overrides model=xgb_classifier            # Override model
mlpipe run --overrides data=higgs_100k model=decision_tree  # Multiple overrides
mlpipe run --overrides model.params.max_depth=8        # Parameter overrides
```

### **Optional Manager CLI (`mlpipe-manager`)**
```bash
mlpipe-manager list                                     # List extras
mlpipe-manager validate                                 # Validate system
mlpipe-manager details model-xgb                       # Show details
mlpipe-manager preview model-xgb evaluation            # Preview install
mlpipe-manager install model-xgb ./my-project          # Install to directory
```

---

## ⚙️ Installation & Dependency Management

### **Development Installation**
```bash
# Full development setup with all dependencies
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates
pip install -e '.[all]'
```

### **Selective Installation**
Install only the dependencies you need:

```bash
# Core framework only
pip install -e '.[core]'

# Traditional ML models
pip install -e '.[xgb,decision-tree,random-forest,svm]'

# Deep learning components
pip install -e '.[torch,gnn,autoencoder]'

# Data science essentials
pip install -e '.[data-csv,data-higgs,preprocessing,evaluation]'
```

### **Available Extras in `pyproject.toml`:**
- **Individual Components:** `model-xgb`, `model-decision-tree`, `model-torch`, etc.
- **Algorithm Groups:** `xgb`, `torch`, `gnn`, `ensemble`
- **Data & Processing:** `data-csv`, `data-higgs`, `preprocessing`, `evaluation`
- **Complete Bundle:** `all` (includes everything)

---

## 📁 Project Structure

```
hep-ml-templates/
├── src/mlpipe/                     # Core library source
│   ├── blocks/                     # Modular components
│   │   ├── model/                  # ML models (traditional + neural)
│   │   ├── ingest/                 # Data loading (CSV, HIGGS, etc.)
│   │   ├── preprocessing/          # Data splitting, scaling, feature eng
│   │   ├── evaluation/             # Metrics and evaluation blocks
│   │   └── training/               # Training orchestration
│   ├── core/                       # Framework interfaces & registry
│   │   ├── interfaces.py           # Base block interfaces
│   │   ├── registry.py             # Component discovery system
│   │   ├── config.py               # Configuration management
│   │   └── utils.py                # Utility functions
│   └── cli/                        # Command-line interfaces
│       ├── main.py                 # `mlpipe` commands
│       ├── manager.py              # `mlpipe-manager` (standalone)
│       └── local_install.py        # Extras installation logic
├── configs/                        # Default YAML configurations
│   ├── model/                      # Model configurations
│   ├── data/                       # Data loader configurations  
│   ├── preprocessing/              # Preprocessing configurations
│   └── pipeline/                   # End-to-end pipeline configurations
├── comprehensive_documentation/    # Complete documentation hub
├── tests/                          # Test suites (unit + integration)
├── pyproject.toml                  # Project metadata, dependencies, CLI entry points
└── README.md                       # This file
```

---

## 🧪 Validation & Testing

### **Comprehensive Validation Results**
- ✅ **6 Core Models Tested:** Decision Tree, Random Forest, XGBoost, SVM, MLP, Ensemble Voting
- ✅ **100% Success Rate:** All models working across different environments
- ✅ **Beginner Testing:** Average setup time <10 seconds per model, rated "extremely easy"
- ✅ **Real-World Integration:** HIGGS benchmark integrated with only 3 line changes
- ✅ **Extras System:** Comprehensive validation across 29 extras with preview/install/validate functionality

### **Production Readiness Indicators**
- 🔍 **Comprehensive Test Suite:** Unit tests, integration tests, end-to-end validation
- 📚 **Complete Documentation:** Master documentation index with guides, reports, and case studies
- 🌐 **Real-World Case Study:** HIGGS100K dataset integration demonstrates practical applicability
- 🔧 **Robust Installation:** Local installation system with dependency management and validation
- ⚡ **Performance Verified:** All models produce expected training/evaluation outputs

---

## 🤝 Contributing

We welcome contributions of new models, datasets, preprocessing utilities, evaluation blocks, and documentation.

### **Adding a New Model**

1. **Implement the Block:**
```python
from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register

@register("model.my_new_model")
class MyNewModel(ModelBlock):
    def build(self, config):
        # Initialize model with config parameters
        pass
    
    def fit(self, X, y):
        # Train the model
        pass
    
    def predict(self, X):
        # Make predictions
        pass
    
    def predict_proba(self, X):  # For classification
        # Return prediction probabilities
        pass
```

2. **Create Configuration:**
```yaml
# configs/model/my_new_model.yaml
block: model.my_new_model
param1: default_value
param2: another_default
random_state: 42
```

3. **Update Extras Mapping:**
Add your model to the extras system in `cli/local_install.py`

4. **Add Tests:**
Create unit tests and integration tests for your model

5. **Update Documentation:**
Add usage examples and update the model list

### **Development Setup**
```bash
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates
pip install -e '.[all]'
# Run tests
python -m pytest tests/ -v
# Validate extras system
mlpipe validate-extras
```

See `CONTRIBUTING.md` for full guidelines, coding standards, and review process.

---

## ❓ FAQ & Troubleshooting

### **Installation Issues**

**Q: Import errors after installation**
```bash
# Ensure you're in the correct directory and installed in editable mode
cd /path/to/your/project
pip install -e .
# Validate the extras system
mlpipe validate-extras
```

**Q: "Model not found" errors**
```bash
# Check what's available
mlpipe list-extras
mlpipe extra-details model-name
# Ensure the model was installed
mlpipe preview-install model-name
```

### **Configuration Questions**

**Q: How do I change hyperparameters without editing YAML files?**
```bash
# Use dotted notation for parameter overrides
mlpipe run --overrides model=xgb_classifier model.params.max_depth=8
mlpipe run --overrides model.params.n_estimators=200 model.params.learning_rate=0.1
```

**Q: How do I combine multiple overrides?**
```bash
# Multiple components and parameters
mlpipe run --overrides data=higgs_100k model=xgb_classifier preprocessing=stratified_split model.params.max_depth=8
```

### **Development Questions**

**Q: How do I preview what components will be installed?**
```bash
# Preview before installing
mlpipe preview-install model-xgb evaluation data-higgs
# Check specific extra contents
mlpipe extra-details model-xgb
```

**Q: How do I validate my installation?**
```bash
# Validate the entire extras system
mlpipe validate-extras
# Test specific functionality
mlpipe list-blocks
mlpipe list-configs
```

---

## 🏆 Research Impact & Applications

### **High Energy Physics Applications**
- **HIGGS Benchmark Integration:** Demonstrated with 3-line code changes, maintaining 100% existing functionality
- **Multi-Model Comparison:** Easy benchmarking across traditional ML and neural network approaches
- **Reproducible Experiments:** Configuration-driven workflows with explicit seeds and consistent data splitting

### **Research Workflow Benefits**
- **Rapid Prototyping:** Test multiple algorithms on the same dataset in minutes
- **Easy Dataset Switching:** Change from demo data to production HIGGS data with single CLI override
- **Collaborative Research:** Share self-contained projects with consistent APIs across teams
- **Paper-Ready Results:** Comprehensive documentation supports research publication requirements

### **Production Deployment**
- **Modular Architecture:** Deploy only the components needed for specific use cases
- **Version Control Friendly:** Configuration-first approach enables clear experiment tracking
- **Scalable Design:** Add new models, datasets, and preprocessing without breaking changes

---

## 📄 License & Acknowledgments

- **License:** MIT License - see `LICENSE` file for details
- **Built On:** Python scientific stack including scikit-learn, XGBoost, pandas, PyTorch, PyTorch Geometric
- **Supported By:** IRIS-HEP fellowship program
- **Community:** Made possible by the High Energy Physics and machine learning communities

### **Citation**
If you use HEP-ML-Templates in your research, please cite:
```bibtex
@software{hep_ml_templates,
  title={HEP-ML-Templates: A Modular Machine Learning Framework for High Energy Physics},
  author={Tawker, Arvind},
  year={2025},
  url={https://github.com/Arvind-t33/hep-ml-templates},
  note={IRIS-HEP Fellowship Project}
}
```

---

## 🚀 Getting Started Now

Ready to start? Here's your path forward:

### **For Quick Experimentation:**
```bash
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates && pip install -e '.[all]'
mlpipe run --overrides model=xgb_classifier
```

### **For New Projects:**
```bash
# In your project directory
mlpipe install-local model-xgb data-higgs evaluation --target-dir .
pip install -e .
mlpipe run
```

### **For Existing Code Integration:**
```bash
# Install specific components
mlpipe install-local model-random-forest preprocessing --target-dir .
# Update imports (see integration examples above)
```

**Questions?** Check the FAQ above, explore `comprehensive_documentation/`, or open an issue on GitHub.

---

*HEP-ML-Templates: Making machine learning in High Energy Physics modular, reproducible, and accessible.*
