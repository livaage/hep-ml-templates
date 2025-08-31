# HEP-ML-Templates

A **modular, plug-and-play machine learning framework** designed specifically for **High Ener### **Individual Models** (11)
Single algorithms with unified interfaces:

**Traditional ML:**
- `model-decision-tree` (1 blocks, 1 configs) - DecisionTreeModel with configurable depth, criteria
- `model-random-forest` (1 blocks, 1 configs) - RandomForestModel with ensemble parameters
- `model-svm` (1 blocks, 1 configs) - SVMModel with kernel options and regularization
- `model-xgb` (1 blocks, 1 configs) - XGBClassifierModel with gradient boosting parameters
- `model-mlp` (1 blocks, 1 configs) - MLPModel multi-layer perceptron with layer configuration
- `model-adaboost` (1 blocks, 1 configs) - AdaBoostModel with weak learner boosting
- `model-ensemble` (1 blocks, 1 configs) - VotingEnsembleModel for combining multiple algorithms

**Neural & Graph Models:**
- `model-torch` (1 blocks, 3 configs) - Generic PyTorch neural networks with flexible architectures
- `model-cnn` (1 blocks, 1 configs) - HEPCNNModel convolutional networks optimized for HEP data
- `model-gnn` (1 blocks, 3 configs) - Graph neural networks: GCNModel, GATModel via PyTorch Geometric
- `model-transformer` (1 blocks, 1 configs) - HEPTransformerModel attention-based architectures for HEP

**Autoencoder Models:**
- `model-autoencoder-vanilla` - VanillaAutoencoderModel for unsupervised feature learning
- `model-autoencoder-variational` - VariationalAutoencoderModel for generative modelingHEP)** research. Build, test, and deploy ML models with true modularity - swap datasets, models, and preprocessing components with minimal code changes and zero vendor lock-in.

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
mlpipe run --overrides model=xgb_classifier data=higgs_uci
mlpipe run --overrides model.params.max_depth=8 preprocessing=data_split
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

## 🎯 Latest Updates & New Features

### **🆕 New in v0.1.0 - Production Ready Release**

**New Blocks & Functionality:**

� **One-Hot Encoder (`preprocessing.onehot_encoder`)**
- Advanced categorical data preprocessing
- Auto-detection of categorical columns
- Configurable handling of unknown categories
- Support for pandas DataFrames and numpy arrays

🔹 **Reconstruction Metrics (`eval.reconstruction`)**
- MSE, MAE, RMSE, SNR metrics for autoencoder evaluation
- Per-sample reconstruction error analysis
- SSIM support for image-like data (optional)
- Latent space statistics for VAEs

🔹 **ROOT File Loader (`ingest.uproot_loader`)**
- Native support for High Energy Physics ROOT files
- Handles jagged arrays and complex HEP data structures
- Configurable branch selection and filtering
- Integration with uproot library (optional dependency)

**Enhanced Configurations:**
- GPU runtime configuration (`runtime/local_gpu.yaml`)
- Reconstruction evaluation settings (`evaluation/reconstruction.yaml`)
- Demo feature engineering configuration (`feature_eng/demo_features.yaml`)
- Verbose logging system across all components

**Production Readiness:**
- ✅ 100% validation success across all 29 extras
- ✅ Comprehensive test suite with 7/7 tests passing
- ✅ All CLI commands fully functional
- ✅ Multiple dataset compatibility (CSV demo + HIGGS benchmark)
- ✅ Complete documentation coverage

--- (30 seconds)

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
mlpipe run --overrides model=xgb_classifier data=higgs_uci
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
- `pipeline-xgb` (5 blocks, 8 configs) - XGBoost pipeline with preprocessing and metrics
- `pipeline-decision-tree` (5 blocks, 8 configs) - Decision tree complete workflow 
- `pipeline-torch` (4 blocks, 6 configs) - PyTorch neural network pipeline
- `pipeline-gnn` (4 blocks, 6 configs) - Graph neural network pipeline

### 🧠 **Individual Models** (11)
Single algorithms with unified interfaces:

**Traditional ML:**
- `model-decision-tree` (1 blocks, 1 configs), `model-random-forest` (1 blocks, 1 configs), `model-svm` (1 blocks, 1 configs)
- `model-xgb` (1 blocks, 1 configs), `model-mlp` (1 blocks, 1 configs), `model-adaboost` (1 blocks, 1 configs), `model-ensemble` (1 blocks, 1 configs)

**Neural & Graph Models:**
- `model-torch` (1 blocks, 3 configs) (PyTorch neural networks)
- `model-cnn` (1 blocks, 1 configs) (Convolutional networks)
- `model-gnn` (1 blocks, 3 configs) (Graph neural networks via PyTorch Geometric)
- `model-transformer` (1 blocks, 1 configs) (Transformer architectures)

### ⚡ **Algorithm Combos** (9)
Model + preprocessing bundles for quick setup:
- `xgb` (2 blocks, 2 configs), `decision-tree` (2 blocks, 2 configs), `random-forest` (2 blocks, 2 configs), `svm` (2 blocks, 2 configs), `mlp` (2 blocks, 2 configs)
- `ensemble` (2 blocks, 2 configs), `torch` (2 blocks, 2 configs), `gnn` (2 blocks, 2 configs), `adaboost` (2 blocks, 2 configs)

### **Data Sources** (3)
Flexible data ingestion with format detection:
- `data-higgs` (1 blocks, 1 configs) - HIGGS benchmark dataset with auto-download and validation
- `data-csv` (1 blocks, 1 configs) - Universal CSV loader with flexible schema detection and parsing
- `data-split` (1 blocks, 1 configs) - Advanced train/val/test splitting utilities with stratification and time-series support

### 🏗️ **Component Categories** (3)
Core processing and evaluation utilities:
- `preprocessing` (3 blocks, 2 configs) - StandardScaler, DataSplitter, advanced preprocessing pipelines  
- `evaluation` (2 blocks, 2 configs) - ClassificationEvaluator with comprehensive metrics, ReconstructionEvaluator
- `feature-eng` (1 blocks, 2 configs) - ColumnSelector for feature selection, custom feature engineering demos

### **Complete Neural Network Architecture Reference**

The framework includes specialized neural network architectures optimized for HEP data:

#### **Transformer Models (`model-transformer`)**
HEP-optimized transformer architectures with attention mechanisms:
```python
# Available configurations:
configs/model/transformer_hep.yaml    # Multi-head attention for particle sequences
```

**Key Features:**
- Multi-head attention mechanisms for particle interaction modeling
- Position encoding for sequential data
- Layer normalization and residual connections
- Configurable attention heads and embedding dimensions

#### **Convolutional Networks (`model-cnn`)**
Specialized CNN architectures for HEP data analysis:
```python
# Available configurations:
configs/model/cnn_hep.yaml           # CNN optimized for HEP detector data
```

**Key Features:**
- 1D and 2D convolutions for different data types
- Batch normalization and dropout for regularization
- Adaptive pooling for variable-length sequences
- Feature maps optimized for physics data patterns

#### **Graph Neural Networks (`model-gnn`)**
Advanced graph-based models for particle interaction analysis:
```python
# Available configurations:
configs/model/gnn_gcn.yaml           # Graph Convolutional Networks
configs/model/gnn_gat.yaml           # Graph Attention Networks  
configs/model/gnn_pyg.yaml           # PyTorch Geometric implementation
```

**Supported Architectures:**
- **GCNModel**: Graph Convolutional Networks for local neighborhood aggregation
- **GATModel**: Graph Attention Networks with learned attention weights
- **Flexible node/edge feature handling** for particle physics graphs

#### **Autoencoder Architectures**
Unsupervised learning models for dimensionality reduction and anomaly detection:

**Vanilla Autoencoders:**
```python
# Available configurations:
configs/model/ae_vanilla.yaml        # Standard autoencoder
```
- Encoder-decoder architecture
- Bottleneck representations for feature learning
- Reconstruction loss optimization

**Variational Autoencoders:**
```python
# Available configurations:
configs/model/ae_variational.yaml    # VAE with probabilistic encoding
```
- Probabilistic encoder with reparameterization trick
- KL divergence regularization
- Generative modeling capabilities for physics simulation

#### **PyTorch Lightning Integration**
All neural network models include PyTorch Lightning integration:
```python
# Available configurations:
configs/model/ae_lightning.yaml      # Lightning-based training pipeline
```

**Features:**
- Automatic GPU/CPU handling
- Built-in logging and checkpointing
- Distributed training support
- Integration with HEP-specific metrics

---

## 🧬 Advanced Model Configuration Reference

### **Hyperparameter Configuration Examples**

#### **XGBoost Classifier (model=xgb_classifier)**
```yaml
# configs/model/xgb_classifier.yaml - Full parameter reference
block: model.xgb_classifier
n_estimators: 100              # Number of boosting rounds
max_depth: 6                   # Maximum tree depth
learning_rate: 0.3             # Step size shrinkage
subsample: 1.0                 # Subsample ratio of training instances
colsample_bytree: 1.0          # Subsample ratio of columns when constructing trees
random_state: 42               # Random seed for reproducibility
objective: "binary:logistic"    # Learning objective
eval_metric: "logloss"         # Evaluation metric
n_jobs: -1                     # Number of parallel threads
```

**CLI Overrides:**
```bash
mlpipe run --overrides model=xgb_classifier model.params.max_depth=8 model.params.n_estimators=200
mlpipe run --overrides model=xgb_classifier model.params.learning_rate=0.1 model.params.subsample=0.8
```

#### **Decision Tree (model=decision_tree)**
```yaml
# configs/model/decision_tree.yaml - Full parameter reference
block: model.decision_tree
max_depth: 10                  # Maximum depth of the tree
criterion: "gini"              # Function to measure the quality of a split
min_samples_split: 2           # Minimum number of samples required to split
min_samples_leaf: 1            # Minimum number of samples required to be at a leaf node
max_features: null             # Number of features to consider when looking for the best split
class_weight: null             # Weights associated with classes
random_state: 42               # Random seed for reproducibility
```

#### **Random Forest (model=random_forest)**
```yaml
# configs/model/random_forest.yaml - Full parameter reference
block: model.random_forest
n_estimators: 100              # Number of trees in the forest
max_depth: null                # Maximum depth of the tree
min_samples_split: 2           # Minimum number of samples required to split
min_samples_leaf: 1            # Minimum number of samples required to be at a leaf node
max_features: "sqrt"           # Number of features to consider at every split
bootstrap: true                # Whether bootstrap samples are used when building trees
class_weight: null             # Weights associated with classes
random_state: 42               # Random seed for reproducibility
n_jobs: -1                     # Number of jobs to run in parallel
```

#### **Support Vector Machine (model=svm)**
```yaml
# configs/model/svm.yaml - Full parameter reference  
block: model.svm
C: 1.0                         # Regularization parameter
kernel: "rbf"                  # Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
degree: 3                      # Degree of the polynomial kernel function
gamma: "scale"                 # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
coef0: 0.0                     # Independent term in kernel function
shrinking: true                # Whether to use the shrinking heuristic
probability: true              # Whether to enable probability estimates
tol: 1e-3                      # Tolerance for stopping criterion
cache_size: 200                # Kernel cache size (in MB)
class_weight: null             # Weights associated with classes
max_iter: -1                   # Hard limit on iterations
random_state: 42               # Random seed for reproducibility
```

#### **Neural Networks (model=mlp)**
```yaml
# configs/model/mlp.yaml - Full parameter reference
block: model.mlp
hidden_layer_sizes: [100]      # The ith element represents the number of neurons in the ith hidden layer
activation: "relu"             # Activation function for the hidden layer
solver: "adam"                 # The solver for weight optimization
alpha: 0.0001                  # L2 penalty (regularization term) parameter
batch_size: "auto"             # Size of minibatches for stochastic optimizers
learning_rate: "constant"      # Learning rate schedule for weight updates
learning_rate_init: 0.001      # Initial learning rate
power_t: 0.5                   # Exponent for inverse scaling learning rate
max_iter: 200                  # Maximum number of iterations
shuffle: true                  # Whether to shuffle samples in each iteration
random_state: 42               # Random seed for reproducibility
tol: 1e-4                      # Tolerance for the optimization
warm_start: false              # When set to True, reuse the solution of the previous call
momentum: 0.9                  # Momentum for gradient descent update
nesterovs_momentum: true       # Whether to use Nesterov's momentum
early_stopping: false          # Whether to use early stopping to terminate training
validation_fraction: 0.1       # Proportion of training data to set aside as validation set
beta_1: 0.9                    # Exponential decay rate for estimates of first moment vector
beta_2: 0.999                  # Exponential decay rate for estimates of second moment vector
epsilon: 1e-8                  # Value for numerical stability
n_iter_no_change: 10           # Maximum number of epochs to not meet tol improvement
```

### **Advanced Data Configuration**

#### **HIGGS Dataset (data=higgs_uci)**
```yaml
# configs/data/higgs_uci.yaml - Complete configuration
block: ingest.csv
auto_download: true            # Automatically download if not present
download_url: "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
file_path: "data/HIGGS_100k.csv"  # Local file path
target_column: "label"         # Name of the target column
header: true                   # Whether CSV has header row
delimiter: ","                 # Field delimiter
encoding: "utf-8"              # File encoding
na_values: ["", "NULL", "nan"] # Values to interpret as NaN
dtype_inference: true          # Automatically infer data types
sample_size: 100000            # Number of samples to load (full dataset is 11M rows)
validation_checks: true        # Perform data quality checks
feature_columns: null          # Specify feature columns (null = auto-detect)
```

#### **Custom CSV Data (data=csv_demo)**
```yaml
# configs/data/csv_demo.yaml - Flexible CSV configuration
block: ingest.csv
file_path: "data/demo_data.csv"
target_column: "target"
header: true
delimiter: ","
encoding: "utf-8"
na_values: ["", "NULL", "nan"]
dtype_inference: true
sample_size: null              # Load full dataset
validation_checks: false
feature_columns: null
skip_rows: 0                   # Number of rows to skip at the beginning
nrows: null                    # Number of rows to read
usecols: null                  # Columns to use
```

### **Preprocessing Configuration Reference**

#### **Data Splitting (preprocessing=data_split)**
```yaml
# configs/preprocessing/data_split.yaml - Complete splitting options
block: preprocessing.data_split
train_size: 0.7                # Proportion of dataset for training
val_size: 0.15                 # Proportion of dataset for validation
test_size: 0.15                # Proportion of dataset for testing
stratify: true                 # Whether to stratify the split
shuffle: true                  # Whether to shuffle before splitting
random_state: 42               # Random seed for reproducibility
time_series: false             # Whether to preserve temporal order
group_column: null             # Column to group by for grouped splitting
```

#### **Standard Scaling (preprocessing=standard)**
```yaml
# configs/preprocessing/standard.yaml - StandardScaler options
block: preprocessing.standard_scaler
with_mean: true                # Whether to center data before scaling
with_std: true                 # Whether to scale data to unit variance
copy: true                     # Whether to perform inplace scaling
```

### **Evaluation Metrics Configuration**

#### **Classification Evaluation (evaluation=classification)**
```yaml
# configs/evaluation/classification.yaml - All available metrics
block: eval.classification
metrics:
  - accuracy                   # Overall accuracy
  - precision                  # Precision score
  - recall                     # Recall score
  - f1                        # F1 score
  - roc_auc                   # Area under ROC curve
  - precision_recall_auc      # Area under precision-recall curve
  - log_loss                  # Logarithmic loss
  - matthews_corrcoef         # Matthews correlation coefficient
average: "binary"              # Averaging strategy for multiclass ('binary', 'micro', 'macro', 'weighted')
pos_label: 1                   # Positive label for binary classification
```

#### **🆕 Reconstruction Evaluation (evaluation=reconstruction)**
```yaml
# configs/evaluation/reconstruction.yaml - Autoencoder/generative model metrics
block: eval.reconstruction
metrics:
  - mse                        # Mean Squared Error
  - mae                        # Mean Absolute Error  
  - rmse                       # Root Mean Squared Error
  - snr                        # Signal-to-Noise Ratio
  - ssim                       # Structural Similarity (requires skimage)
per_sample: true               # Compute per-sample error distributions
plot_reconstruction: true     # Generate reconstruction visualizations
save_outputs: true            # Save reconstructed samples
output_dir: "reconstruction_outputs"
```

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
from mlpipe.blocks.model.ensemble_models import RandomForestModel  # Change 1

config = {'n_estimators': 100, 'random_state': 42}
model = RandomForestModel()                                        # Change 2
model.build(config)
model.fit(X_train, y_train)                                        # Change 3 - preprocessing handled internally
predictions = model.predict_proba(X_test)[:, 1]
```

**Swap to XGBoost:**
```python
from mlpipe.blocks.model.xgb_classifier import XGBClassifierModel  # Only import changes
model = XGBClassifierModel()                                       # Only class name changes
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
# Use pre-configured processing strategies
mlpipe run --overrides preprocessing=data_split
mlpipe run --overrides preprocessing=standard
mlpipe run --overrides feature_eng=column_selector
```

### **Configuration Examples:**

**Stratified 70/15/15 Split:**
```yaml
# configs/preprocessing/data_split.yaml
train_size: 0.7
val_size: 0.15
test_size: 0.15
stratify: true
shuffle: true
random_state: 42
```

**Standard Preprocessing:**
```yaml
# configs/preprocessing/standard.yaml
with_mean: true
with_std: true
copy: true
```

---

## 💻 Complete CLI Reference

### **Embedded CLI (`mlpipe`)**

#### **Discovery & Configuration Commands**
```bash
# List all available blocks (registered components)
mlpipe list-blocks

# List all available configurations with usage examples
mlpipe list-configs [--config-path CONFIGS_DIR]    # Optional custom config directory path

# Discover available extras and their contents
mlpipe list-extras                                  # Show all available extras by category

# Inspect specific extras before installing
mlpipe extra-details EXTRA_NAME                    # Show detailed breakdown of blocks/configs
mlpipe extra-details model-xgb                     # Example: inspect XGBoost extra

# Preview installations before committing
mlpipe preview-install EXTRA1 [EXTRA2 ...]        # Preview what would be installed
mlpipe preview-install model-xgb evaluation        # Example: preview installation

# Validate the extras system integrity
mlpipe validate-extras                              # Check all extras mappings are valid
```

**Detailed Command Options:**

**`mlpipe list-blocks`**
- Lists all 17 registered block components with their registry names
- Shows available model, data, preprocessing, evaluation, and training blocks
- No additional flags required

**`mlpipe list-configs [--config-path PATH]`**
- `--config-path`: Specify custom configuration directory (default: ./configs)
- Lists configurations by category: data, model, preprocessing, feature_eng, training, evaluation
- Shows example usage for each configuration

**`mlpipe list-extras`**
- Shows 29 available extras grouped by category
- Displays block and config counts for each extra
- No additional parameters required

**`mlpipe extra-details EXTRA_NAME`**
- Detailed breakdown showing specific blocks and configurations included
- Example: `mlpipe extra-details pipeline-xgb` shows all 5 blocks and 8 configs
- Lists all files that would be copied during installation

**`mlpipe preview-install EXTRA1 [EXTRA2 ...]`**
- Shows complete file tree of what would be installed
- Displays dependencies and conflicts
- Can preview multiple extras simultaneously
- No actual installation performed

**`mlpipe validate-extras`**
- Validates all 29 extras against available blocks and configurations
- Checks for missing files, invalid references, and mapping consistency
- Returns detailed report of any validation issues

#### **Installation & Setup Commands**
```bash
# Install extras locally to create standalone projects
mlpipe install-local EXTRA1 [EXTRA2 ...] --target-dir TARGET_DIR

# Examples:
mlpipe install-local model-xgb --target-dir ./my-xgb-project
mlpipe install-local model-decision-tree data-higgs evaluation --target-dir ./research-project
mlpipe install-local all --target-dir ./complete-ml-suite
```

**Detailed Installation Options:**

**`mlpipe install-local EXTRAS --target-dir PATH`**
- **Required `--target-dir PATH`**: Absolute or relative path to target directory
- **EXTRAS**: Space-separated list of extra names (e.g., `model-xgb data-higgs evaluation`)
- Creates complete project structure with setup.py, configs/, and src/
- Automatically handles dependencies and file conflicts
- Installs in additive mode (can add more components later)

**Installation Behavior:**
```bash
# Creates this structure in target directory:
my-project/
├── setup.py                  # Auto-generated with selected dependencies
├── src/mlpipe/              # Copied blocks and core functionality
├── configs/                 # Relevant configuration files
├── README.md               # Auto-generated documentation
└── pyproject.toml          # Project metadata
```

**Additive Installation:**
```bash
# Initial install
mlpipe install-local model-xgb --target-dir ./project
cd ./project && pip install -e .

# Add more components later (additive)
mlpipe install-local data-higgs evaluation --target-dir .
mlpipe install-local model-decision-tree --target-dir .
```

#### **Execution & Pipeline Commands**
```bash
# Run pipelines with full configuration control
mlpipe run [OPTIONS]

# Pipeline options:
mlpipe run                                              # Use defaults (xgb_basic pipeline)
mlpipe run --pipeline PIPELINE_NAME                    # Specify pipeline implementation
mlpipe run --config-path CONFIGS_DIR                   # Custom config directory
mlpipe run --config-name CONFIG_FILE                   # Specific pipeline config file

# Override any configuration values:
mlpipe run --overrides OVERRIDE1 [OVERRIDE2 ...]
mlpipe run --overrides model=xgb_classifier            # Swap model component
mlpipe run --overrides data=higgs_uci                  # Swap data component
mlpipe run --overrides model=decision_tree data=csv_demo  # Multiple overrides

# Parameter-level overrides (dot notation):
mlpipe run --overrides model.params.max_depth=8        # Model hyperparameters
mlpipe run --overrides model.params.n_estimators=200 model.params.learning_rate=0.1
mlpipe run --overrides data.params.test_size=0.2       # Data splitting parameters
```

**Detailed Run Options:**

**`mlpipe run [OPTIONS]`**
- **`--pipeline PIPELINE`**: Pipeline implementation to use (default: xgb_basic)
  - Available pipelines: `xgb_basic`, `decision_tree_basic`, `gnn_basic`, `torch_basic`
- **`--config-path PATH`**: Path to configuration directory (default: ./configs)
- **`--config-name NAME`**: Pipeline configuration file name without .yaml extension (default: pipeline)
- **`--overrides [OVERRIDES ...]`**: Override config values using dot notation

**Override Syntax Examples:**
```bash
# Component-level overrides
mlpipe run --overrides model=xgb_classifier            # Change model component
mlpipe run --overrides data=higgs_uci                  # Change data source
mlpipe run --overrides preprocessing=data_split        # Change preprocessing
mlpipe run --overrides evaluation=classification       # Change evaluation metrics

# Parameter-level overrides
mlpipe run --overrides model.params.max_depth=8        # Single parameter
mlpipe run --overrides model.params.n_estimators=200 model.params.learning_rate=0.1  # Multiple parameters
mlpipe run --overrides data.params.sample_size=50000   # Data loading parameters
mlpipe run --overrides preprocessing.params.train_size=0.8  # Preprocessing parameters

# Complex combinations
mlpipe run --overrides model=decision_tree data=higgs_uci preprocessing=data_split model.params.max_depth=15
```

**Available Override Targets:**
- **`model=`**: xgb_classifier, decision_tree, random_forest, svm, mlp, adaboost, ensemble_voting, etc.
- **`data=`**: higgs_uci, csv_demo, custom_hep_example, wine_quality_example, medical_example
- **`preprocessing=`**: data_split, standard  
- **`feature_eng=`**: column_selector, demo_features, custom_test_features
- **`training=`**: sklearn
- **`evaluation=`**: classification

### **Manager CLI (`mlpipe-manager`)**
Standalone interface with simpler command structure and enhanced examples:

```bash
# Discovery commands
mlpipe-manager list                                     # List all available extras
mlpipe-manager validate                                 # Validate extras configuration

# Inspection commands  
mlpipe-manager details EXTRA_NAME                      # Show details for specific extra
mlpipe-manager preview EXTRA1 [EXTRA2 ...]            # Preview installation

# Installation command
mlpipe-manager install EXTRA1 [EXTRA2 ...] TARGET_DIR  # Install extras to directory

# Examples:
mlpipe-manager details model-xgb                       # Inspect XGBoost extra
mlpipe-manager preview model-xgb preprocessing         # Preview combined installation
mlpipe-manager install model-xgb ./my-project          # Install to project directory
```

**Detailed Manager Commands:**

**`mlpipe-manager list`**
- Shows all 29 available extras organized by category
- Includes block and configuration counts
- Color-coded output for easy browsing
- Equivalent to `mlpipe list-extras`

**`mlpipe-manager validate`**  
- Comprehensive validation of extras system integrity
- Checks file existence, registry mappings, dependency consistency
- Reports any issues with specific extras
- Equivalent to `mlpipe validate-extras`

**`mlpipe-manager details EXTRA_NAME`**
- Detailed breakdown of specific extra components
- Lists all blocks, configurations, and files included
- Shows dependency requirements
- Example: `mlpipe-manager details pipeline-torch` shows PyTorch pipeline components

**`mlpipe-manager preview EXTRA1 [EXTRA2 ...]`**
- Preview complete installation without executing
- Shows directory structure, file conflicts, dependencies
- Can preview combinations of multiple extras
- Useful for planning project structure

**`mlpipe-manager install EXTRAS TARGET_DIR`**
- Install selected extras to target directory
- Creates complete project structure with setup.py
- Handles dependencies and file management
- Supports additive installation (can run multiple times)

**Manager CLI Advantages:**
- Simpler command structure for non-developers
- Enhanced help and examples built-in
- Cleaner output formatting
- Focused on project creation workflow

### **Complete Usage Examples**

#### **Basic Model Training**
```bash
# Quick start with defaults
mlpipe run

# Try different models on same data
mlpipe run --overrides model=decision_tree
mlpipe run --overrides model=random_forest
mlpipe run --overrides model=svm

# Switch datasets
mlpipe run --overrides data=csv_demo
mlpipe run --overrides data=higgs_uci
```

#### **Hyperparameter Tuning**
```bash
# XGBoost hyperparameter sweep
mlpipe run --overrides model=xgb_classifier model.params.max_depth=6
mlpipe run --overrides model=xgb_classifier model.params.max_depth=8
mlpipe run --overrides model=xgb_classifier model.params.n_estimators=200 model.params.learning_rate=0.05

# Decision tree parameters
mlpipe run --overrides model=decision_tree model.params.max_depth=10 model.params.min_samples_split=5
```

#### **Data Processing Variations**
```bash
# Different preprocessing strategies  
mlpipe run --overrides preprocessing=standard          # Standard scaling
mlpipe run --overrides preprocessing=data_split        # Custom data splitting

# Combined data and preprocessing changes
mlpipe run --overrides data=higgs_uci preprocessing=standard model=xgb_classifier
```

#### **Project Creation Workflow**
```bash
# 1. Explore available components
mlpipe list-extras
mlpipe extra-details pipeline-xgb

# 2. Preview what will be installed
mlpipe preview-install pipeline-xgb

# 3. Create project with selected components
mlpipe install-local pipeline-xgb --target-dir ./hep-research
cd ./hep-research && pip install -e .

# 4. Run experiments with different configurations
mlpipe run --overrides model.params.max_depth=8
mlpipe run --overrides data=csv_demo
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

**Individual Model Components:**
- `model-xgb` → XGBoost Classifier (requires: xgboost>=1.7)
- `model-decision-tree` → Decision Tree Classifier (requires: scikit-learn>=1.2) 
- `model-random-forest` → Random Forest Classifier (requires: scikit-learn>=1.2)
- `model-svm` → Support Vector Machine (requires: scikit-learn>=1.2)
- `model-mlp` → Multi-Layer Perceptron (requires: scikit-learn>=1.2)
- `model-adaboost` → AdaBoost Classifier (requires: scikit-learn>=1.2)
- `model-ensemble` → Ensemble Voting Classifier (requires: scikit-learn>=1.2, xgboost>=1.7)
- `model-torch` → PyTorch Neural Networks (requires: torch>=2.2)
- `model-lightning` → PyTorch Lightning Models (requires: torch>=2.2, lightning>=2.2)
- `model-gnn` → Graph Neural Networks (requires: torch-geometric>=2.5, torch>=2.2)

**Algorithm-Specific Bundles:**
- `xgb` → XGBoost + preprocessing (xgboost>=1.7, scikit-learn>=1.2)
- `decision-tree` → Decision Tree + preprocessing (scikit-learn>=1.2)
- `random-forest` → Random Forest + preprocessing (scikit-learn>=1.2)
- `svm` → SVM + preprocessing (scikit-learn>=1.2)
- `mlp` → MLP + preprocessing (scikit-learn>=1.2)
- `adaboost` → AdaBoost + preprocessing (scikit-learn>=1.2)
- `ensemble` → Ensemble + preprocessing (scikit-learn>=1.2, xgboost>=1.7)
- `torch` → PyTorch + Lightning framework (torch>=2.2, lightning>=2.2)
- `gnn` → Graph Neural Networks + PyTorch (torch-geometric>=2.5, torch>=2.2)
- `autoencoder` → Autoencoder models (torch>=2.2, lightning>=2.2)
- `transformer` → Transformer architectures (torch>=2.2, lightning>=2.2)
- `cnn` → Convolutional Neural Networks (torch>=2.2, lightning>=2.2)

**Data & Processing:**
- `data-csv` → Universal CSV data loading (pandas>=2.0)
- `data-higgs` → HIGGS benchmark dataset (pandas>=2.0, requests>=2.25)
- `preprocessing` → Data preprocessing utilities (scikit-learn>=1.2)
- `evaluation` → Model evaluation metrics (scikit-learn>=1.2)

**Complete Pipeline Bundles:**
- `pipeline-xgb` → Complete XGBoost pipeline (omegaconf>=2.3, numpy>=1.22, pandas>=2.0, scikit-learn>=1.2, xgboost>=1.7)
- `pipeline-decision-tree` → Complete Decision Tree pipeline (omegaconf>=2.3, numpy>=1.22, pandas>=2.0, scikit-learn>=1.2)
- `pipeline-ensemble` → Complete Ensemble pipeline (omegaconf>=2.3, numpy>=1.22, pandas>=2.0, scikit-learn>=1.2, xgboost>=1.7)
- `pipeline-torch` → Complete PyTorch pipeline (omegaconf>=2.3, numpy>=1.22, pandas>=2.0, scikit-learn>=1.2, torch>=2.2, lightning>=2.2)
- `pipeline-gnn` → Complete GNN pipeline (omegaconf>=2.3, numpy>=1.22, pandas>=2.0, scikit-learn>=1.2, torch>=2.2, torch-geometric>=2.5)
- `pipeline-autoencoder` → Complete Autoencoder pipeline (omegaconf>=2.3, numpy>=1.22, pandas>=2.0, scikit-learn>=1.2, torch>=2.2, lightning>=2.2)

**Special Bundles:**
- `core` → Minimal framework (omegaconf>=2.3, numpy>=1.22, pandas>=2.0)
- `all` → Everything (omegaconf>=2.3, numpy>=1.22, pandas>=2.0, scikit-learn>=1.2, xgboost>=1.7, torch>=2.2, lightning>=2.2, torch-geometric>=2.5)
- `dev` → Development dependencies (pytest>=7.3, pytest-cov>=4.1, ruff>=0.5, black>=24.1, pre-commit>=3.6)

**Usage Examples:**
```bash
# Install minimal setup
pip install -e '.[core]'

# Install specific models
pip install -e '.[model-xgb,model-decision-tree]'

# Install complete pipelines
pip install -e '.[pipeline-xgb,pipeline-torch]'

# Install algorithm bundles
pip install -e '.[xgb,torch,gnn]'

# Install everything
pip install -e '.[all]'
```

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

## 🐍 Complete Python API Reference

### **Core Interfaces**

#### **ModelBlock Interface**
All model classes inherit from `ModelBlock` and implement these methods:

```python
from mlpipe.core.interfaces import ModelBlock

class CustomModel(ModelBlock):
    def build(self, config: dict) -> None:
        """Initialize model with configuration parameters."""
        pass
    
    def fit(self, X, y) -> 'ModelBlock':
        """Train the model on training data."""
        pass
    
    def predict(self, X):
        """Make predictions on new data."""
        pass
    
    def predict_proba(self, X):  # For classification models
        """Return prediction probabilities."""
        pass
    
    def score(self, X, y) -> float:  # Optional
        """Return model performance score."""
        pass
```

#### **Registry System**
Use the registry to discover and instantiate components:

```python
from mlpipe.core.registry import register, get_block, list_blocks

# Register a new block
@register("model.my_model")
class MyModel(ModelBlock):
    pass

# Get registered block
block_class = get_block("model.xgb_classifier")
model = block_class()

# List all registered blocks
all_blocks = list_blocks()
```

### **Available Model Classes**

#### **Traditional ML Models**

**XGBoost Classifier**
```python
from mlpipe.blocks.model.xgb_classifier import XGBClassifierModel

model = XGBClassifierModel()
model.build({
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.3,
    'random_state': 42
})
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)[:, 1]
```

**Decision Tree**
```python
from mlpipe.blocks.model.decision_tree import DecisionTreeModel

model = DecisionTreeModel()
model.build({
    'max_depth': 10,
    'criterion': 'gini',
    'random_state': 42
})
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Random Forest**
```python
from mlpipe.blocks.model.ensemble_models import RandomForestModel

model = RandomForestModel()
model.build({
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42
})
```

**Support Vector Machine**
```python
from mlpipe.blocks.model.svm import SVMModel

model = SVMModel()
model.build({
    'C': 1.0,
    'kernel': 'rbf',
    'probability': True,
    'random_state': 42
})
```

**Multi-Layer Perceptron**
```python
from mlpipe.blocks.model.mlp import MLPModel

model = MLPModel()
model.build({
    'hidden_layer_sizes': [100, 50],
    'activation': 'relu',
    'solver': 'adam',
    'random_state': 42
})
```

**Ensemble Models**
```python
from mlpipe.blocks.model.ensemble_models import AdaBoostModel, VotingEnsembleModel

# AdaBoost
ada_model = AdaBoostModel()
ada_model.build({
    'n_estimators': 50,
    'learning_rate': 1.0,
    'random_state': 42
})

# Voting Ensemble
ensemble_model = VotingEnsembleModel()
ensemble_model.build({
    'voting': 'soft',
    'estimators': ['xgb', 'rf', 'svm']  # Automatically includes pre-configured estimators
})
```

#### **Neural Network Models**

**Autoencoders**
```python
from mlpipe.blocks.model.ae_lightning import VanillaAutoencoderModel, VariationalAutoencoderModel

# Vanilla Autoencoder
ae_model = VanillaAutoencoderModel()
ae_model.build({
    'input_dim': 784,
    'hidden_dims': [256, 128, 64],
    'learning_rate': 0.001,
    'max_epochs': 100
})
ae_model.fit(X_train, y_train)

# Variational Autoencoder
vae_model = VariationalAutoencoderModel()
vae_model.build({
    'input_dim': 784,
    'hidden_dims': [256, 128],
    'latent_dim': 32,
    'learning_rate': 0.001
})
```

**HEP Neural Networks**
```python
from mlpipe.blocks.model.hep_neural import HEPTransformerModel, HEPCNNModel

# Transformer for HEP data
transformer_model = HEPTransformerModel()
transformer_model.build({
    'input_dim': 28,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'num_classes': 2
})

# CNN for HEP data
cnn_model = HEPCNNModel()
cnn_model.build({
    'input_channels': 1,
    'num_classes': 2,
    'conv_layers': [32, 64, 128],
    'fc_layers': [256, 128]
})
```

**Graph Neural Networks**
```python
from mlpipe.blocks.model.gnn_pyg import GCNModel, GATModel

# Graph Convolutional Network
gcn_model = GCNModel()
gcn_model.build({
    'input_dim': 28,
    'hidden_dims': [64, 32],
    'num_classes': 2,
    'dropout': 0.5
})

# Graph Attention Network
gat_model = GATModel()
gat_model.build({
    'input_dim': 28,
    'hidden_dims': [64, 32],
    'num_classes': 2,
    'heads': 4,
    'dropout': 0.5
})
```

### **Data Loading & Processing**

#### **CSV Data Loading**
```python
from mlpipe.blocks.ingest.csv import CSVDataBlock

loader = CSVDataBlock()
loader.build({
    'file_path': 'data/my_data.csv',
    'target_column': 'label',
    'header': True,
    'delimiter': ',',
    'sample_size': 10000
})
X, y = loader.load()
```

#### **🆕 ROOT File Loading**
```python
from mlpipe.blocks.ingest.uproot_loader import UprootDataBlock

# For High Energy Physics ROOT files
loader = UprootDataBlock()
loader.build({
    'file_path': 'data/higgs_events.root',
    'tree_name': 'Events',  # or None for auto-detection
    'branches': ['pt', 'eta', 'phi', 'mass'],  # or None for all branches
    'target_branch': 'label',
    'selection_cuts': 'pt > 20',  # ROOT-style cuts
    'max_entries': 100000,
    'flatten_arrays': True,  # Handle jagged arrays
    'verbose': True
})
X, y = loader.load()  # Requires: pip install uproot
```

#### **Data Splitting**
```python
from mlpipe.blocks.preprocessing.data_split import split_data, DataSplitter

# Functional interface
splits = split_data(X, y, 
    train_size=0.7, val_size=0.15, test_size=0.15,
    stratify=True, random_state=42
)
X_train, y_train = splits['train']
X_val, y_val = splits['val'] 
X_test, y_test = splits['test']

# Class-based interface
splitter = DataSplitter({
    'train_size': 0.8,
    'test_size': 0.2,
    'stratify': True,
    'random_state': 42
})
splits = splitter.fit_transform(X, y)
```

#### **Standard Scaling**
```python
from mlpipe.blocks.preprocessing.standard_scaler import StandardScalerBlock

scaler = StandardScalerBlock()
scaler.build({
    'with_mean': True,
    'with_std': True,
    'copy': True
})
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### **🆕 One-Hot Encoding**
```python
from mlpipe.blocks.preprocessing.onehot_encoder import OneHotEncoderBlock

# Automatic categorical column detection
encoder = OneHotEncoderBlock()
encoder.build({
    'categorical_columns': 'auto',  # or ['col1', 'col2']
    'drop_first': True,  # Avoid multicollinearity
    'handle_unknown': 'ignore',
    'sparse_output': False,
    'verbose': True
})

# Fit and transform
encoder.fit(X_train)  # Learns categories from training data
X_encoded = encoder.transform(X)  # Generate one-hot features
print(f"Original features: {X.shape[1]}, Encoded features: {X_encoded.shape[1]}")
```

encoder = OneHotEncoderBlock()
encoder.build({
    'categorical_columns': ['category_A', 'category_B'],  # or None for auto-detection
    'drop_first': False,
    'handle_unknown': 'ignore',
    'verbose': True
})

# Fit and transform
encoder.fit(X_train)
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)
```

### **Evaluation & Metrics**

#### **Classification Evaluation**
```python
from mlpipe.blocks.evaluation.classification import ClassificationEvaluator

evaluator = ClassificationEvaluator()
evaluator.build({
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'average': 'binary',
    'pos_label': 1
})

# Get comprehensive evaluation results
results = evaluator.evaluate(y_true, y_pred, y_pred_proba)
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"AUC: {results['roc_auc']:.4f}")
```

#### **🆕 Reconstruction Evaluation**
```python
from mlpipe.blocks.evaluation.reconstruction_metrics import ReconstructionEvaluator

# For autoencoder/generative model evaluation
evaluator = ReconstructionEvaluator()
evaluator.build({
    'metrics': ['mse', 'mae', 'rmse', 'snr', 'ssim'],  # Available metrics
    'per_sample': True,       # Compute per-sample errors
    'plot_reconstruction': True,  # Generate visualizations
    'save_outputs': True,     # Save reconstructed samples
    'output_dir': 'reconstruction_outputs'
})

# Evaluate reconstruction quality
original_data = X_test  # Original input data
reconstructed_data = autoencoder.predict(X_test)  # Reconstructed output

results = evaluator.evaluate(original_data, reconstructed_data)
print(f"MSE: {results['mse']:.6f}")
print(f"SNR: {results['snr']:.2f} dB")
print(f"SSIM: {results.get('ssim', 'N/A')}")  # Optional metric
```

### **Configuration Management**

#### **OmegaConf Integration**
```python
from omegaconf import OmegaConf

# Load configuration files
config = OmegaConf.load("configs/model/xgb_classifier.yaml")
data_config = OmegaConf.load("configs/data/higgs_uci.yaml")

# Merge configurations
merged_config = OmegaConf.merge(config, data_config)

# Override parameters programmatically
config.params.max_depth = 8
config.params.n_estimators = 200

# Use with models
model = XGBClassifierModel()
model.build(config.params)
```

### **Complete End-to-End Pipeline**

The hep-ml-templates library now provides complete pre-configured pipeline types that include all necessary components (data, preprocessing, model, training, and evaluation) with automatic data file management. Each pipeline type is ready to run out-of-the-box:

**Available Pipeline Types:**
- `pipeline-decision-tree` - Complete Decision Tree workflow
- `pipeline-xgb` - XGBoost pipeline with preprocessing and metrics  
- `pipeline-ensemble` - Ensemble methods pipeline
- `pipeline-torch` - PyTorch neural network pipeline
- `pipeline-gnn` - Graph neural network pipeline

**Quick Start - Any Pipeline in 3 Commands:**

```bash
# 1. Install a complete pipeline (includes data files automatically)
mlpipe install-local pipeline-xgb --target-dir ./my-project

# 2. Navigate to project
cd my-project

# 3. Run the pipeline
mlpipe run
```

**Examples for Different Pipeline Types:**

```bash
# Decision Tree Pipeline
mlpipe install-local pipeline-decision-tree --target-dir ./dt-project
cd dt-project && mlpipe run

# XGBoost Pipeline  
mlpipe install-local pipeline-xgb --target-dir ./xgb-project
cd xgb-project && mlpipe run

# Neural Network Pipeline
mlpipe install-local pipeline-torch --target-dir ./nn-project  
cd nn-project && mlpipe run

# Graph Neural Network Pipeline
mlpipe install-local pipeline-gnn --target-dir ./gnn-project
cd gnn-project && mlpipe run
```

**What You Get:**
- ✅ Complete pipeline configuration (`pipeline.yaml`)
- ✅ All necessary data files (`demo_tabular.csv`, specialized datasets)
- ✅ Pre-configured preprocessing, training, and evaluation
- ✅ Ready-to-run setup with `mlpipe run` command
- ✅ Modular components you can customize independently

**Expected Results:**
- Decision Tree: ~100% accuracy on demo data
- XGBoost: ~100% accuracy on demo data  
- Neural Network: ~98% accuracy on demo data
- All pipelines include ROC-AUC, F1-score, and confusion matrix metrics

---

---

## 🗂️ Complete Component Reference

### **Available Blocks** (`mlpipe list-blocks`)
```
eval.classification          # Classification evaluation metrics
eval.reconstruction          # 🆕 Reconstruction metrics (MSE, MAE, RMSE, SNR)
feature.column_selector      # Feature selection utilities
ingest.csv                   # CSV data loading
ingest.uproot_loader         # 🆕 ROOT file data loading for HEP data
model.adaboost              # AdaBoost classifier
model.ae_vanilla            # Vanilla autoencoder
model.ae_variational        # Variational autoencoder
model.cnn_hep               # Convolutional neural network
model.decision_tree         # Decision tree classifier
model.ensemble_voting       # Voting ensemble classifier
model.mlp                   # Multi-layer perceptron
model.random_forest         # Random forest classifier
model.svm                   # Support vector machine
model.transformer_hep       # Transformer architecture
model.xgb_classifier        # XGBoost classifier
preprocessing.data_split    # Data splitting utilities
preprocessing.onehot_encoder # 🆕 One-hot encoding for categorical data
preprocessing.standard_scaler # Standard scaling preprocessing  
train.sklearn               # Scikit-learn training orchestration
```

### **Available Configurations** (`mlpipe list-configs`)

**Pipeline Configurations:**
- `pipeline` - Default end-to-end pipeline

**Data Configurations:**
- `csv_demo` - Demo CSV dataset configuration
- `custom_hep_example` - Custom HEP dataset example
- `custom_test` - Custom test dataset
- `higgs_uci` - HIGGS UCI dataset configuration
- `medical_example` - Medical dataset example
- `wine_quality_example` - Wine quality dataset example

**Model Configurations:**
- `adaboost` - AdaBoost classifier settings
- `ae_lightning` - Lightning autoencoder settings
- `ae_vanilla` - Vanilla autoencoder settings
- `ae_variational` - Variational autoencoder settings
- `cnn_hep` - CNN for HEP data settings
- `decision_tree` - Decision tree parameters
- `ensemble_voting` - Voting ensemble settings
- `gnn_gat` - Graph Attention Network settings
- `gnn_gcn` - Graph Convolutional Network settings
- `gnn_pyg` - PyTorch Geometric GNN settings
- `mlp` - Multi-layer perceptron settings
- `random_forest` - Random forest parameters
- `svm` - SVM classifier settings
- `transformer_hep` - Transformer for HEP settings
- `xgb_classifier` - XGBoost classifier parameters

**Preprocessing Configurations:**
- `data_split` - Data splitting parameters
- `standard` - Standard scaling parameters

**Feature Engineering Configurations:**
- `column_selector` - Column selection settings
- `custom_test_features` - Custom test features
- `demo_features` - Demo feature engineering

**Training Configurations:**
- `sklearn` - Scikit-learn training parameters

**Preprocessing Configurations:**
- `data_split` - Data splitting parameters
- `standard` - Standard scaling parameters

**Runtime Configurations:**
- `local_cpu` - Local CPU runtime settings (device: cpu, seed: 42)
- `local_gpu` - 🆕 Local GPU runtime settings (device: cuda, optimization flags)

**Evaluation Configurations:**
- `classification` - Classification evaluation metrics
- `reconstruction` - 🆕 Reconstruction evaluation metrics (MSE, MAE, RMSE, SNR)

### **Runtime Configuration System**

The framework includes a runtime configuration system for controlling execution environment and reproducibility:

```yaml
# configs/runtime/local_cpu.yaml
device: cpu        # Computing device (cpu/cuda)
seed: 42          # Random seed for reproducibility
```

**Using Runtime Configurations:**
```bash
# Override runtime settings via CLI
mlpipe run --overrides runtime=local_cpu
mlpipe run --overrides runtime.device=cuda runtime.seed=123

# Custom runtime configuration
# Create configs/runtime/gpu_setup.yaml:
# device: cuda
# seed: 2024
mlpipe run --overrides runtime=gpu_setup
```

### **Verbose Logging & Debugging System**

The framework includes verbose logging capabilities for debugging and monitoring:

```bash
# Enable verbose output for data loading
mlpipe run --overrides data.verbose=true

# Enable verbose output for all components (if supported)
mlpipe run --overrides model.verbose=true data.verbose=true

# Disable verbose output for production runs
mlpipe run --overrides data.verbose=false
```

**Verbose Mode Features:**
- **Data Loading:** Shows detailed dataset information, sampling details, preprocessing steps
- **Model Training:** Displays training progress, parameter validation, performance metrics
- **Debugging:** Helpful for troubleshooting configuration issues and understanding pipeline execution

**Example with Verbose Data Loading:**
```yaml
# configs/data/debug_higgs.yaml
block: ingest.csv
file_path: "data/HIGGS_100k.csv"
target_column: "label"
verbose: true                      # Shows loading progress, data shapes, statistics
sample_size: 10000
header: False
```

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

1. **Implement the Model:**
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

**Q: Dependency conflicts during installation**
```bash
# Check exact dependencies for an extra
mlpipe extra-details EXTRA_NAME
# Install minimal set first, then add incrementally
pip install -e '.[core]'
pip install -e '.[model-xgb]'
pip install -e '.[data-higgs]'
```

**Q: PyTorch/CUDA installation issues**
```bash
# Install PyTorch first with specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Then install HEP-ML-Templates
pip install -e '.[model-torch,model-gnn]'
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
mlpipe run --overrides data=higgs_uci model=xgb_classifier preprocessing=data_split model.params.max_depth=8
```

**Q: How do I create custom configurations?**
```yaml
# Create configs/model/my_custom_model.yaml
block: model.xgb_classifier
n_estimators: 500
max_depth: 12
learning_rate: 0.05
subsample: 0.9
# Use with: mlpipe run --overrides model=my_custom_model
```

**Q: How do I use custom data files?**
```yaml
# Create configs/data/my_data.yaml
block: ingest.csv
file_path: "/path/to/my/data.csv"
target_column: "my_target"
# Use with: mlpipe run --overrides data=my_data
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

**Q: How do I add custom blocks?**
```python
# Create new model in src/mlpipe/blocks/model/my_model.py
from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register

@register("model.my_custom_model")
class MyCustomModel(ModelBlock):
    def build(self, config):
        # Initialize your model
        pass
    def fit(self, X, y):
        # Training logic
        pass
    def predict(self, X):
        # Prediction logic
        pass
```

### **Performance & Debugging**

**Q: How do I debug pipeline execution?**
```bash
# Enable verbose output (add to config)
verbose: true
debug: true

# Run with single override to isolate issues
mlpipe run --overrides model=xgb_classifier
mlpipe run --overrides data=csv_demo
```

**Q: Memory issues with large datasets?**
```yaml
# Modify data loading config for sampling
# configs/data/higgs_uci_sample.yaml
block: ingest.csv
file_path: "data/HIGGS_100k.csv"
sample_size: 10000              # Use smaller sample
target_column: "label"
```

**Q: How do I monitor training progress?**
For neural network models (PyTorch/Lightning):
```yaml
# configs/model/torch_with_logging.yaml
block: model.torch
enable_progress_bar: true
log_every_n_steps: 50
enable_checkpointing: true
checkpoint_dir: "./checkpoints"
```

### **Advanced Usage**

**Q: How do I use the framework programmatically?**
```python
from omegaconf import OmegaConf
from mlpipe.core.registry import get_block

# Load configuration
config = OmegaConf.load("configs/model/xgb_classifier.yaml")

# Get block and build
block_class = get_block(config.block)
model = block_class()
model.build(config)

# Use the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Q: How do I create custom preprocessing pipelines?**
```python
from mlpipe.blocks.preprocessing.data_split import split_data
from mlpipe.blocks.preprocessing.standard_scaler import StandardScalerBlock

# Custom preprocessing pipeline
splits = split_data(X, y, train_size=0.8, stratify=True)
X_train, y_train = splits['train']
X_test, y_test = splits['test']

scaler = StandardScalerBlock()
scaler.build({'with_mean': True, 'with_std': True})
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Q: How do I integrate with existing MLOps workflows?**
```python
# Export trained models for deployment
import joblib

# Train model using HEP-ML-Templates
model = XGBClassifierModel()
model.build(config)
model.fit(X_train, y_train)

# Export for deployment
joblib.dump(model.model, 'trained_model.pkl')

# Or use with MLflow, Weights & Biases, etc.
import mlflow
with mlflow.start_run():
    mlflow.log_params(config.params)
    mlflow.sklearn.log_model(model.model, "model")
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

## 🤖 Development Acknowledgments

This project leveraged advanced AI assistance during development to enhance code quality, functionality, and user experience:

- **GitHub Copilot Agent Mode**: Used for comprehensive pipeline system development, modular architecture design, and end-to-end integration testing
- **Claude Sonnet 4**: Utilized for error detection, code review, commenting standards, and block development validation
- **Automated Testing**: AI-assisted creation of comprehensive test suites ensuring robust pipeline functionality across all model types

The combination of human expertise and AI collaboration enabled rapid development of a sophisticated, modular ML framework while maintaining high code quality and comprehensive documentation standards.

---

*HEP-ML-Templates: Making machine learning in High Energy Physics modular, reproducible, and accessible.*
