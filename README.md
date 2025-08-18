# HEP ML Templates

A modular machine learning pipeline framework designed for High Energy Physics data analysis.

## Features

- **Modular Architecture**: Pluggable blocks for data ingestion, preprocessing, feature engineering, and model training
- **Registry System**: Automatic block discovery and registration
- **HEP-Focused**: Built-in support for common HEP datasets (HIGGS UCI dataset)
- **Configurable Pipelines**: YAML-based configuration system
- **Multiple Algorithms**: Support for XGBoost and extensible to other ML algorithms

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hep-ml-templates.git
cd hep-ml-templates

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Running a Pipeline

```bash
# Run with default configuration (HIGGS dataset)
mlpipe run

# Easy modularity - switch datasets with one command
mlpipe run --overrides data=csv_demo feature_eng=demo_features

# Switch back to HIGGS dataset  
mlpipe run --overrides data=higgs_uci

# Mix and match any components
mlpipe run --overrides data=csv_demo model=xgb_classifier preprocessing=standard

# List all available configurations
mlpipe list-configs

# List available pipeline blocks
mlpipe list-blocks
```

### Key Features: True Modularity

The system provides **two complementary approaches** for modularity:

#### 🚀 **Quick Overrides** (for experiments)
- **Change dataset**: `--overrides data=higgs_uci` or `data=csv_demo`
- **Change model**: `--overrides model=xgb_classifier` 
- **Change preprocessing**: `--overrides preprocessing=standard`
- **Mix multiple**: `--overrides data=csv_demo model=xgb_classifier`

#### 🎛️ **Pipeline Configuration** (for persistent changes)
- **Edit once, use everywhere**: Modify `configs/pipeline.yaml` to set new defaults
- **Component-specific configs**: Add datasets in `configs/data/`, models in `configs/model/`, etc.
- **Single source of truth**: `pipeline.yaml` orchestrates all components by referencing their configs
- **Share configurations**: Easy to version control and share complete pipeline setups

No need to create new YAML files - just override the components you want to change!

## Tutorials

### Tutorial 1: Quick Component Swapping (Using Overrides)

The fastest way to experiment with different components:

```bash
# Start with default HIGGS dataset
mlpipe run

# Switch to demo dataset with matching features  
mlpipe run --overrides data=csv_demo feature_eng=demo_features

# Try different combinations
mlpipe run --overrides data=csv_demo model=xgb_classifier preprocessing=standard

# See what's available
mlpipe list-configs
```

**When to use**: Quick experiments, testing different combinations, one-off runs.

### Tutorial 2: Persistent Configuration Changes (Editing configs)

For permanent changes or new default setups, edit the main pipeline orchestrator:

#### Step 1: Check available components
```bash
mlpipe list-configs
```

#### Step 2: Edit the pipeline orchestrator
Open `configs/pipeline.yaml`:
```yaml
data: higgs_uci              # ← Change this to csv_demo
preprocessing: standard
feature_eng: column_selector # ← Change this to demo_features  
model: xgb_classifier
training: sklearn
evaluation: classification
runtime: local_cpu
```

#### Step 3: Run with new defaults
```bash
mlpipe run  # Now uses your chosen components as defaults
```

**When to use**: Setting new defaults, sharing configurations, production setups.

### Tutorial 3: Adding a New Dataset

Want to add your own dataset? Follow this pattern:

#### Step 1: Create dataset configuration
Create `configs/data/my_dataset.yaml`:
```yaml
block: ingest.csv
path: data/my_data.csv
label: target
has_header: true
names: null  # or specify column names
label_is_first_column: false
nrows: null  # or limit rows
```

#### Step 2: Update pipeline to use it
Edit `configs/pipeline.yaml`:
```yaml
data: my_dataset  # ← Points to your new configs/data/my_dataset.yaml
preprocessing: standard
feature_eng: column_selector
model: xgb_classifier
training: sklearn
evaluation: classification
runtime: local_cpu
```

#### Step 3: Run your pipeline
```bash
mlpipe run  # Uses your new dataset
```

#### Step 4 (Optional): Create matching feature engineering
If your dataset has different columns, create `configs/feature_eng/my_features.yaml`:
```yaml
block: feature.column_selector
include: [col1, col2, col3]  # Your column names
exclude: []
```

Then update `configs/pipeline.yaml`:
```yaml
data: my_dataset
feature_eng: my_features  # ← Points to your custom features
# ... rest stays the same
```

## Adding a New Component to the Pipeline

To extend the pipeline with a new block (e.g. dataset, feature engineering step, training routine, or model), follow this general process:

1. **Check if it already exists**

   ```bash
   mlpipe list-configs
   ```

   Look under the relevant section (e.g. `model`, `dataset`, `feature_eng`, `training`).

2. **If not present, implement the component**

   * Add the code to the correct folder under `src/mlpipe/blocks/<component>/`.
     For example, a new model goes in:

     ```
     src/mlpipe/blocks/model/<your_model>.py
     ```

3. **Create a configuration file**

   * Add a YAML file describing your component to:

     ```
     configs/<component>/<your_component>.yaml
     ```

4. **Modify the pipeline definition**

   * Open `pipeline.yaml`.
   * Update the relevant section (e.g. `model:`) to point to your new component.
   * Ensure the **name** you use here is consistent with your code and config file.

---

💡 **Note**: The same workflow applies for switching out any block — whether `dataset`, `feature_eng`, `training`, or `model`. Just stay consistent with naming across **code**, **configs**, and **pipeline.yaml**.

---

**Key principle**: `pipeline.yaml` is your single control panel. It orchestrates everything by pointing to the right component configs in `configs/data/`, `configs/model/`, `configs/feature_eng/`, etc.

## Configuration Architecture

The framework uses a **hierarchical configuration system** designed for maximum modularity:

### 📋 Pipeline Level (`configs/pipeline.yaml`)
The **single source of truth** that orchestrates your entire pipeline:
```yaml
data: higgs_uci              # → loads configs/data/higgs_uci.yaml
preprocessing: standard      # → loads configs/preprocessing/standard.yaml  
feature_eng: column_selector # → loads configs/feature_eng/column_selector.yaml
model: xgb_classifier        # → loads configs/model/xgb_classifier.yaml
training: sklearn            # → loads configs/training/sklearn.yaml
evaluation: classification   # → loads configs/evaluation/classification.yaml
```

### 🧩 Component Level (`configs/*/`)
Each component has its own configuration space:
- `configs/data/` - Dataset configurations (paths, formats, preprocessing)
- `configs/model/` - Model architectures and hyperparameters  
- `configs/preprocessing/` - Data preprocessing steps
- `configs/feature_eng/` - Feature engineering configurations
- `configs/training/` - Training procedures and parameters
- `configs/evaluation/` - Evaluation metrics and methods

### 🔄 Two Ways to Change Components
1. **Temporary** (overrides): `mlpipe run --overrides data=csv_demo`
2. **Persistent** (edit pipeline.yaml): Change `data: higgs_uci` → `data: csv_demo`

This design ensures **clean separation of concerns** - each component configuration focuses on its specific domain, while the pipeline configuration handles orchestration.

## Project Structure

```
hep-ml-templates/
├── src/mlpipe/
│   ├── blocks/           # Modular pipeline components
│   │   ├── ingest/       # Data ingestion blocks
│   │   ├── preprocess/   # Preprocessing blocks
│   │   ├── feature_eng/  # Feature engineering blocks
│   │   ├── model/        # ML model blocks
│   │   ├── training/     # Training blocks
│   │   └── evaluation/   # Evaluation blocks
│   ├── core/             # Core framework components
│   ├── pipelines/        # Complete pipeline implementations
│   ├── cli/              # Command-line interface
│   └── templates/        # Pipeline templates
├── configs/              # YAML configuration files
│   ├── data/            # Dataset configurations
│   ├── model/           # Model configurations
│   ├── preprocessing/   # Preprocessing configurations
│   ├── feature_eng/     # Feature engineering configurations
│   ├── training/        # Training configurations
│   └── evaluation/      # Evaluation configurations
├── examples/            # Example pipeline implementations
│   ├── xgb_basic/       # XGBoost classification example
│   ├── ae_basic/        # Autoencoder example
│   └── gnn_basic/       # Graph Neural Network example
├── data/                # Dataset storage
├── docs/                # Documentation
└── tests/               # Unit and integration tests
```

## Supported Datasets

- **HIGGS UCI Dataset**: Particle physics classification task
- Extensible to other HEP datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here]
