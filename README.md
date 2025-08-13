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
# Run XGBoost pipeline on HIGGS dataset
mlpipe --pipeline xgb_basic --config-path configs --config-name higgs_uci
```

## Project Structure

```
hep-ml-templates/
├── src/mlpipe/
│   ├── blocks/           # Modular pipeline components
│   │   ├── ingest/       # Data ingestion blocks
│   │   ├── preprocess/   # Preprocessing blocks
│   │   └── feature_eng/  # Feature engineering blocks
│   ├── core/             # Core framework components
│   ├── pipelines/        # Complete pipeline implementations
│   └── cli/              # Command-line interface
├── configs/              # YAML configuration files
└── data/                 # Dataset storage
```

## Supported Datasets

- **HIGGS UCI Dataset**: Particle physics classification task
- Extensible to other HEP datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here]
