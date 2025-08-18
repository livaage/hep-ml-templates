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

The system is designed for **easy component swapping**:

- **Change dataset**: `--overrides data=higgs_uci` or `data=csv_demo`
- **Change model**: `--overrides model=xgb_classifier`
- **Change preprocessing**: `--overrides preprocessing=standard`
- **Change feature engineering**: `--overrides feature_eng=column_selector`
- **Mix multiple**: `--overrides data=csv_demo model=xgb_classifier`

No need to create new YAML files - just override the components you want to change!

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
