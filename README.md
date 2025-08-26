# HEP ML Templates

A **modular machine learning pipeline framework** designed for High Energy Physics data analysis. Build, test, and deploy ML models with true plug-and-play modularity - swap datasets, models, and preprocessing components with a single command.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🧩 **True Modularity**: Mix and match components - datasets, models, preprocessing - independently
- 🎯 **HEP-Optimized**: Built specifically for particle physics data and workflows
- ⚡ **Zero-Config Start**: Works out of the box with sensible defaults
- 🔧 **Selective Installation**: Install only the components you need
- 🚀 **CLI-First**: Powerful command-line interface for rapid experimentation
- 📊 **Multi-Algorithm**: XGBoost, Decision Trees, with easy extensibility to PyTorch, GNNs, and more

---

# 📖 Usage Guide

Perfect for researchers, data scientists, and HEP analysts who want to rapidly prototype and experiment with different ML approaches on particle physics data.

## 🚀 Quick Start (30 seconds)

This is currently a development library. Here's how to get started:

### Step 1: Get the Code
```bash
# Clone or download the repository
git clone https://github.com/YOUR_USERNAME/hep-ml-templates.git
cd hep-ml-templates
```

### Step 2: Install with Dependencies
```bash
# Install everything you need
pip install -e '.[all]'

# Verify installation
mlpipe --help
```

### Step 3: Run Your First Pipeline
```bash
# Run immediately - configs are already here!
mlpipe run

# Expected output:
# ✅ Loading complete: Features: (100000, 28), Target: (100000,)
# === Metrics ===
# auc: 0.6886
# accuracy: 0.6357
```

**That's it!** No config copying, no path issues, no complex setup.

## Installation Options

All installation options assume you've cloned the repository first.

### Complete Installation (Recommended)
```bash
cd hep-ml-templates
pip install -e '.[all]'
```

### Selective Installation (Choose What You Need)
```bash
cd hep-ml-templates

# Just XGBoost pipeline
pip install -e '.[model-xgb,data-higgs,preprocessing]'

# Just Decision Tree pipeline  
pip install -e '.[model-decision-tree,data-higgs,preprocessing]'

# Core framework only
pip install -e '.'
```

### Virtual Environment (Recommended for Isolation)
```bash
cd hep-ml-templates
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e '.[all]'
```

> **macOS Note**: XGBoost requires OpenMP runtime. Install with: `brew install libomp`

## 🔄 Component Swapping

Since you're in the repository directory, all configs are immediately available:

```bash
# Switch to demo dataset (perfect for testing)
mlpipe run --overrides data=csv_demo feature_eng=demo_features
# Output: AUC: 1.0000, Accuracy: 1.0000 (300 samples, 3 features)

# Switch models (keep same data)
mlpipe run --overrides model=decision_tree  
# Output: AUC: 0.6779, Accuracy: 0.6245 (HIGGS + Decision Tree)

# Mix and match anything
mlpipe run --overrides data=csv_demo model=decision_tree feature_eng=demo_features
# Output: AUC: 1.0000, Accuracy: 1.0000 (Demo + Decision Tree)

# See all available options
mlpipe list-configs
```

## 🔍 Explore Available Components

```bash
# See all available configurations
mlpipe list-configs

# See all registered blocks  
mlpipe list-blocks

# Get help for any command
mlpipe run --help
```

---

# 📚 Tutorials

## Tutorial 1: Dataset Swapping

**Goal**: Test the same model on different datasets to compare performance.

```bash
# Test on HIGGS dataset (large, challenging)
mlpipe run --overrides data=higgs_uci
# Typical output: AUC: 0.6886, Accuracy: 0.6357

# Test on demo dataset (small, easy)  
mlpipe run --overrides data=csv_demo feature_eng=demo_features
# Typical output: AUC: 1.0000, Accuracy: 1.0000

# Compare results easily
echo "=== HIGGS Dataset ===" && mlpipe run --overrides data=higgs_uci | grep -E "(auc|accuracy)"
echo "=== Demo Dataset ===" && mlpipe run --overrides data=csv_demo feature_eng=demo_features | grep -E "(auc|accuracy)"
```

## Tutorial 2: Model Comparison

**Goal**: Compare different ML algorithms on the same dataset.

```bash
# XGBoost baseline (default)
mlpipe run --overrides model=xgb_classifier
# Typical output: AUC: 0.6886, Accuracy: 0.6357

# Decision Tree comparison
mlpipe run --overrides model=decision_tree  
# Typical output: AUC: 0.6779, Accuracy: 0.6245

# Test both on demo data for quick comparison
mlpipe run --overrides data=csv_demo feature_eng=demo_features model=xgb_classifier
mlpipe run --overrides data=csv_demo feature_eng=demo_features model=decision_tree
```

## Tutorial 3: Adding Your Own Dataset

**Goal**: Use your own data with the pipeline.

### Step 1: Prepare Your Data
Create a CSV file with clear column names:
```csv
feature1,feature2,feature3,signal_label
1.2,2.5,0.8,signal
0.9,1.1,2.2,background
1.5,3.1,1.2,signal
0.7,1.8,2.8,background
```

### Step 2: Create Dataset Configuration
```bash
# Create your dataset config
cat > configs/data/my_dataset.yaml << EOF
block: ingest.csv
file_path: "data/my_data.csv"
target_column: "signal_label"
has_header: true
EOF
```

### Step 3: Create Feature Configuration (Optional)
```bash
# Create feature engineering config
cat > configs/feature_eng/my_features.yaml << EOF
block: feature.column_selector
include: ["feature1", "feature2", "feature3"]
exclude: []
EOF
```

### Step 4: Test Your Dataset
```bash
# Make sure your data file is in the data/ directory
cp /path/to/your/data.csv data/my_data.csv

# Test with XGBoost
mlpipe run --overrides data=my_dataset feature_eng=my_features model=xgb_classifier

# Test with Decision Tree
mlpipe run --overrides data=my_dataset feature_eng=my_features model=decision_tree
```

---

# 🔧 Development & Contribution Guide

Perfect for developers who want to extend the framework with new models, datasets, or preprocessing components.

## Framework Architecture

The framework is built around **modular blocks** that implement standard interfaces:

```
src/mlpipe/
├── blocks/              # 🧩 All modular components
│   ├── ingest/         # Data loading blocks
│   ├── model/          # ML model blocks  
│   ├── preprocess/     # Data preprocessing blocks
│   ├── feature_eng/    # Feature engineering blocks
│   └── evaluation/     # Evaluation blocks
├── core/               # 🏗️ Framework core
│   ├── interfaces.py   # Block contracts/interfaces
│   └── registry.py     # Auto-discovery system
└── cli/               # 💻 Command-line interface
```

## Adding a New Model (Beginner Tutorial)

Let's add a simple **Linear Regression** model step by step.

### Step 1: Create the Model File

Create `src/mlpipe/blocks/model/linear_regression.py`:

```python
from sklearn.linear_model import LogisticRegression
from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register

@register("model.linear_regression")
class LinearRegressionBlock(ModelBlock):
    """Simple logistic regression model."""
    
    def __init__(self, **kwargs):
        # Default parameters
        default_params = {
            'max_iter': 1000,
            'random_state': 42
        }
        self.params = {**default_params, **kwargs}
        self.model = None
        
    def build(self, config=None):
        """Build the model with parameters."""
        if config:
            params = {**self.params, **config}
        else:
            params = self.params
            
        self.model = LogisticRegression(**params)
        print(f"✅ Linear Regression built with max_iter={params['max_iter']}")
        
    def fit(self, X, y):
        """Train the model."""
        if self.model is None:
            self.build()  # Auto-build with defaults
            
        self.model.fit(X, y)
        print("✅ Linear Regression training completed!")
        
    def predict(self, X):
        """Make predictions (return probabilities for binary classification)."""
        return self.model.predict_proba(X)[:, 1]
```

### Step 2: Create Configuration File

Create `configs/model/linear_regression.yaml`:

```yaml
# Linear Regression Configuration
block: model.linear_regression

# Parameters
max_iter: 1000
random_state: 42

# Documentation:
# max_iter: Maximum number of iterations for solver convergence
# random_state: Random seed for reproducible results
```

### Step 3: Register the Model

Add to `src/mlpipe/blocks/model/__init__.py`:

```python
# Existing imports...
try:
    from . import xgb_classifier
except ImportError:
    pass

try:
    from . import decision_tree  
except ImportError:
    pass

try:
    from . import linear_regression    # ← Add this line
except ImportError:
    pass  # scikit-learn not available
```

### Step 4: Test Your Model

```bash
# Test the new model
mlpipe run --overrides model=linear_regression

# Test on demo data for quick results
mlpipe run --overrides data=csv_demo model=linear_regression feature_eng=demo_features

# Compare with other models
echo "=== XGBoost ===" && mlpipe run --overrides model=xgb_classifier | grep -E "(auc|accuracy)"
echo "=== Decision Tree ===" && mlpipe run --overrides model=decision_tree | grep -E "(auc|accuracy)"  
echo "=== Linear Regression ===" && mlpipe run --overrides model=linear_regression | grep -E "(auc|accuracy)"
```

### Step 5: Verify It's Available

```bash
# Check if your model is registered
mlpipe list-blocks | grep linear_regression

# Check if config is available
mlpipe list-configs | grep linear_regression
```

**Congratulations!** You've added a new model to the framework! 🎉

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the HEP ML community's need for reproducible, modular ML pipelines
- Powered by scikit-learn, XGBoost, pandas, and numpy
- Thank you to all contributors and beta testers!

---

**Happy ML modeling! 🚀**

# Get specific block
model_class = get("model.xgb_classifier")
model_instance = model_class(config={'max_depth': 5})

# Use the block
model_instance.build()
model_instance.fit(X_train, y_train)
predictions = model_instance.predict(X_test)
```

### 3. Configuration Loading

Configs are loaded hierarchically:

```python
# pipeline.yaml says: model: xgb_classifier  
# This loads configs/model/xgb_classifier.yaml
# Which says: block: model.xgb_classifier
# Registry provides: XGBClassifierBlock class
```

## Block Interfaces

All blocks must implement specific interfaces. Here are the key ones:

### ModelBlock Interface

```python
from mlpipe.core.interfaces import ModelBlock

class YourModelBlock(ModelBlock):
    def build(self, config=None):
        """Initialize the model with config."""
        pass
        
    def fit(self, X, y):
        """Train the model."""
        pass
        
    def predict(self, X):
        """Make predictions (probabilities for classification)."""
        pass
        
    def predict_classes(self, X):
        """Make class predictions (optional)."""
        pass
```

### DataBlock Interface

```python
from mlpipe.core.interfaces import DataBlock

class YourDataBlock(DataBlock):
    def load(self):
        """Load and return X, y, metadata."""
        return X, y, metadata
```

### PreprocessingBlock Interface

```python
from mlpipe.core.interfaces import PreprocessingBlock

class YourPreprocessingBlock(PreprocessingBlock):
    def transform(self, X, y=None, metadata=None):
        """Transform data and return X, y, metadata."""
        return X_transformed, y, metadata
```

## Testing Your Components

### Unit Testing

Create tests for your blocks:

```python
# tests/unit/test_my_model.py
import pytest
import numpy as np
from mlpipe.core.registry import get

def test_my_model_basic():
    """Test basic model functionality."""
    # Get your model from registry
    model_class = get("model.my_model")
    model = model_class()
    
    # Create dummy data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Test training
    model.fit(X, y)
    
    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert 0 <= predictions.min() <= predictions.max() <= 1

def test_my_model_config():
    """Test model configuration."""
    config = {'param1': 'value1'}
    model = get("model.my_model")(config=config)
    model.build()
    # Test that config was applied correctly
```

### Integration Testing

Test your component with the full pipeline:

```python
# tests/integration/test_my_component_integration.py
import subprocess

def test_my_model_integration():
    """Test model works with full pipeline."""
    cmd = "mlpipe run --overrides model=my_model data=csv_demo feature_eng=demo_features"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "auc:" in result.stdout
    assert "accuracy:" in result.stdout
```

### Manual Testing

Use the CLI for rapid testing:

```bash
# Test your block exists
mlpipe list-blocks | grep my_model

# Test basic functionality
mlpipe run --overrides model=my_model

# Test with different data
mlpipe run --overrides model=my_model data=csv_demo feature_eng=demo_features

# Test configuration override
mlpipe run --overrides model=my_model model.param1=new_value
```

## Best Practices

### 1. Configuration Design

- **Provide sensible defaults**: Your block should work with minimal config
- **Document all parameters**: Use comments in YAML configs
- **Validate inputs**: Check required fields and types
- **Handle missing dependencies gracefully**: Use try/except for imports

```yaml
# Good config example
block: model.my_model

# Core parameters (required)
param1: default_value

# Optional parameters with good defaults
param2: auto  # auto-detect best value
param3: null  # disable this feature

# Documentation
# param1: Controls the main behavior (values: A, B, C)  
# param2: Optimization strategy (auto|manual|disabled)
# param3: Additional feature (null to disable)
```

### 2. Error Handling

```python
@register("model.robust_model")
class RobustModelBlock(ModelBlock):
    def __init__(self, config=None):
        self.config = config or {}
        
        # Validate required dependencies
        try:
            import required_library
        except ImportError:
            raise ImportError(
                "required_library is needed for RobustModel. "
                "Install with: pip install 'hep-ml-templates[my-extra]'"
            )
            
    def fit(self, X, y):
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValueError("Training data cannot be empty")
            
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
            
        # Proceed with training...
```

### 3. Logging and User Feedback

```python
def fit(self, X, y):
    print(f"🔧 Training {self.__class__.__name__} with {len(X)} samples...")
    
    # Show progress for long operations
    if len(X) > 10000:
        print("⏱️ Large dataset detected, training may take a while...")
    
    self.model.fit(X, y)
    
    print(f"✅ Training completed! Model ready for predictions.")
```

### 4. Extensibility

Design your blocks to be easily extended:

```python
@register("model.base_neural_network")
class BaseNeuralNetworkBlock(ModelBlock):
    """Base class for neural network models."""
    
    def __init__(self, **kwargs):
        self.params = self.get_default_params()
        self.params.update(kwargs)
        
    def get_default_params(self):
        """Override this in subclasses."""
        return {'learning_rate': 0.001}
        
    def build_architecture(self):
        """Override this to define model architecture."""
        raise NotImplementedError
        
@register("model.custom_cnn")        
class CustomCNNBlock(BaseNeuralNetworkBlock):
    """Specific CNN implementation."""
    
    def get_default_params(self):
        params = super().get_default_params()
        params.update({
            'conv_layers': [32, 64],
            'kernel_size': 3
        })
        return params
        
    def build_architecture(self):
        # CNN-specific architecture
        pass
```

## Contributing Guidelines

### Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hep-ml-templates.git
   cd hep-ml-templates
   ```

2. **Create Development Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/my-new-model
   ```

4. **Develop and Test**
   ```bash
   # Add your code
   # Write tests
   pytest tests/
   
   # Test with real pipelines
   mlpipe run --overrides model=my_new_model
   ```

5. **Format and Lint**
   ```bash
   black src/ tests/
   ruff src/ tests/
   ```

6. **Submit Pull Request**

### Code Standards

- **Python Style**: Black formatting, max line length 100
- **Documentation**: Docstrings for all public functions and classes
- **Testing**: Unit tests for all new functionality, integration tests for new blocks
- **Type Hints**: Use type hints where helpful
- **Error Messages**: Clear, actionable error messages with installation instructions

### Commit Messages

Use conventional commits:

```
feat: add neural network model support
fix: handle missing data in CSV loader  
docs: improve installation instructions
test: add integration tests for model swapping
```

## 📊 Performance and Scaling

### Benchmarking Your Components

```python
# Example benchmark script
import time
import numpy as np
from mlpipe.core.registry import get

def benchmark_model(model_name, data_sizes=[1000, 10000, 100000]):
    """Benchmark model training time vs dataset size."""
    model_class = get(f"model.{model_name}")
    
    results = {}
    for n in data_sizes:
        # Generate synthetic data
        X = np.random.randn(n, 50) 
        y = np.random.randint(0, 2, n)
        
        # Time training
        model = model_class()
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()
        
        results[n] = end_time - start_time
        print(f"{model_name} - {n:6d} samples: {results[n]:.2f}s")
    
    return results

# Benchmark different models
benchmark_model("xgb_classifier")
benchmark_model("decision_tree")
benchmark_model("neural_network")
```

### Memory Management

For large datasets:

```python
@register("ingest.chunked_csv")
class ChunkedCSVLoader(DataBlock):
    """CSV loader with chunked processing for large files."""
    
    def load(self):
        chunk_size = self.config.get('chunk_size', 10000)
        file_path = self.config['file_path']
        
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Process chunk if needed
            processed_chunk = self.process_chunk(chunk)
            chunks.append(processed_chunk)
            
        # Combine chunks
        data = pd.concat(chunks, ignore_index=True)
        # ... rest of processing
```

## 🤝 Community and Support

### Getting Help

- **Documentation**: Start with `docs/` folder
- **Examples**: Check `examples/` for working code
- **Issues**: Search existing [GitHub Issues](https://github.com/YOUR_USERNAME/hep-ml-templates/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/YOUR_USERNAME/hep-ml-templates/discussions)

### Contributing Ideas

Looking for ways to contribute? Here are areas where help is especially welcome:

**🤖 Models**
- PyTorch/Lightning integration
- Transformer models for HEP
- Graph Neural Networks (GNNs)
- Autoencoder variants
- Ensemble methods

**📊 Data Handling**
- ROOT file support (uproot integration)
- Parquet format support  
- HDF5 data loading
- Data streaming for large datasets
- Multi-file dataset handling

**🔧 Preprocessing**
- HEP-specific transformations
- Feature selection algorithms
- Data augmentation techniques
- Outlier detection
- Normalization strategies

**⚡ Performance**
- GPU acceleration
- Distributed training
- Model optimization
- Caching strategies
- Parallel processing

**🧪 Testing**
- More comprehensive test datasets
- Performance benchmarks
- Cross-validation utilities
- Model comparison tools

**📚 Documentation**
- Tutorials for specific use cases
- Video walkthroughs
- API documentation
- Best practices guides

### Recognition

Contributors are recognized in:
- README acknowledgments
- Release notes
- Conference presentations
- Academic papers (where appropriate)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the HEP ML community's need for reproducible, modular ML pipelines
- Built on the shoulders of giants: scikit-learn, XGBoost, pandas, numpy
- Special thanks to all contributors and beta testers

## 📞 Contact

- **Maintainer**: [Your Name](mailto:your.email@example.com)
- **Project**: [GitHub Repository](https://github.com/YOUR_USERNAME/hep-ml-templates)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/YOUR_USERNAME/hep-ml-templates/issues)
