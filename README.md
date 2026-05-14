# hep-ml-templates

A modular machine-learning pipeline framework for High Energy Physics. Pipelines are assembled from small "blocks" (ingest, preprocessing, model, training, evaluation) wired together by YAML configs. Install only the pieces you need, or scaffold a self-contained project that you can edit and version freely.

Requires Python 3.10+.

## Install

Clone and install in editable mode:

```bash
git clone https://github.com/livaage/hep-ml-templates.git
cd hep-ml-templates
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[core]"
```

Add one or more *extras* to pull in algorithm-specific dependencies:

```bash
pip install -e ".[pipeline-xgb]"          # complete XGBoost pipeline
pip install -e ".[pipeline-gnn]"          # complete GNN pipeline
pip install -e ".[xgb,data-higgs]"        # mix individual extras
pip install -e ".[all]"                   # everything
```

The full list of extras is in [`pyproject.toml`](pyproject.toml). The categories are:

- **Pipeline bundles** — `pipeline-xgb`, `pipeline-decision-tree`, `pipeline-ensemble`, `pipeline-neural`, `pipeline-gnn`, `pipeline-autoencoder-lightning`
- **Algorithms** — `xgb`, `decision-tree`, `random-forest`, `svm`, `mlp`, `adaboost`, `ensemble`, `torch`, `gnn`, `autoencoder`, `transformer`, `cnn`
- **Block groups** — `data-csv`, `data-higgs`, `data-uproot`, `model-*`, `preprocessing`, `evaluation`
- **Dev** — `dev` (pytest, ruff, black, mypy, pre-commit)

`scripts/install.sh` is a thin wrapper if you prefer not to type pip incantations:

```bash
scripts/install.sh pipeline-xgb
scripts/install.sh xgb data-higgs evaluation
```

## Quick start

Run an end-to-end pipeline against the bundled HIGGS_100k sample:

```bash
pip install -e ".[pipeline-xgb]"
mlpipe run --overrides data=higgs_uci model=xgb_classifier
```

Try swapping just the model or the data:

```bash
mlpipe run --overrides model=random_forest
mlpipe run --overrides data=csv_demo model=mlp
```

Scaffold a standalone project that you can edit without touching the library:

```bash
mlpipe install-local pipeline-xgb --target-dir ./my-project
cd my-project
pip install -e .
mlpipe run
```

The scaffold copies the block code and configs for the chosen extras into `./my-project/mlpipe/` and `./my-project/configs/`. The `mlpipe` CLI prefers local blocks over the installed package, so changes you make in the project take effect immediately.

## CLI

`mlpipe` provides the runtime and scaffolding:

| Command | Purpose |
|---|---|
| `mlpipe run` | Run a pipeline from `configs/pipeline.yaml` (override with `--overrides k=v`) |
| `mlpipe list-blocks` | List registered blocks |
| `mlpipe list-configs` | List YAML configs under `configs/` |
| `mlpipe list-extras` | List pip extras declared in `pyproject.toml` |
| `mlpipe extra-details <extra>` | Show what an extra installs |
| `mlpipe preview-install <extras…>` | Dry-run an `install-local` |
| `mlpipe install-local <extras…> --target-dir DIR` | Copy blocks and configs into `DIR` |
| `mlpipe generate-pipeline <type>` | Write a starter `pipeline.yaml` for a given pipeline type |
| `mlpipe validate-config` | Validate a pipeline config |
| `mlpipe pipeline-info` | Inspect a pipeline config |

`mlpipe-manager` is an alternative front-end focused on extras discovery (`list`, `details`, `validate`, `preview`); use whichever fits your workflow.

## Architecture

```
configs/
  pipeline.yaml           # top-level config — picks one block per slot
  data/*.yaml             # data loader options
  preprocessing/*.yaml
  feature_eng/*.yaml
  model/*.yaml
  training/*.yaml
  evaluation/*.yaml
  runtime/*.yaml          # device, seed, logging

src/mlpipe/
  core/                   # registry, config loading, universal runner
  blocks/
    ingest/               # csv_loader, uproot_loader, graph_csv_loader
    preprocessing/        # standard_scaler, data_split, onehot_encoder
    feature_eng/          # column_selector
    model/                # xgb_classifier, decision_tree, ensemble_models, mlp, svm,
                          # hep_neural (Transformer/CNN), gnn_pyg, ae_lightning
    training/             # sklearn_trainer, pytorch_trainer
    evaluation/           # classification_metrics, reconstruction_metrics
  cli/                    # main + manager entry points
```

Each block registers itself with the central registry via `@register("category.name")`. The config file names a block per slot (e.g. `model: xgb_classifier`), and the runner builds and connects them. To add a new block, write a class implementing the relevant interface in `src/mlpipe/core/interfaces.py`, decorate it with `@register(...)`, and drop a YAML config under `configs/<category>/`.

## Configuration

Configs are YAML loaded by OmegaConf, so dotted-path overrides work on the CLI:

```bash
mlpipe run --overrides model=xgb_classifier model.n_estimators=500 data=higgs_uci
```

A top-level `configs/pipeline.yaml` selects one config from each category. To swap, edit `pipeline.yaml` or use `--overrides`. To add new defaults for a block, write a new `configs/<category>/<name>.yaml` file.

You can also drive the pipeline from Python:

```python
from mlpipe.core.config import load_config
from mlpipe.core.universal_runner import run_pipeline

cfg = load_config("configs", "pipeline", overrides=["model=xgb_classifier"])
run_pipeline(config_path="configs", config_name="pipeline",
             overrides=["model=xgb_classifier"])
```

Individual blocks can be instantiated directly:

```python
from mlpipe.blocks.model.xgb_classifier import XGBClassifierBlock

model = XGBClassifierBlock(n_estimators=200, max_depth=6)
model.build()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Bundled data

`data/` contains small files committed for demos:

- `HIGGS_100k.csv` — first 100k rows of the UCI HIGGS dataset
- `demo_tabular.csv` — generic tabular demo (also used by the test suite)
- `graph_nodes_demo.csv` — minimal graph data for GNN demos

For larger HIGGS data, configure `data/higgs_uci.yaml` to point at the full file; the loader will download it on demand.

## Available blocks

A quick reference — run `mlpipe list-blocks` for the authoritative list.

**Ingest:** `ingest.csv`, `ingest.uproot_loader`, `ingest.graph_csv`
**Preprocessing:** `preprocessing.standard_scaler`, `preprocessing.data_split`, `preprocessing.onehot_encoder`
**Feature engineering:** `feature.column_selector`
**Models:** `model.xgb_classifier`, `model.decision_tree`, `model.random_forest`, `model.adaboost`, `model.ensemble_voting`, `model.svm`, `model.mlp`, `model.transformer_hep`, `model.cnn_hep`, `model.gnn_gcn`, `model.gnn_gat`, `model.ae_vanilla`, `model.ae_variational`
**Training:** `train.sklearn`, `train.pytorch`
**Evaluation:** `eval.classification`, `eval.reconstruction`

Some blocks (e.g. `model.xgb_classifier`, `ingest.uproot_loader`) only register themselves once their optional dependency is importable. If you don't see one, install the matching extra.

## GNN install note

The `gnn` / `pipeline-gnn` extras depend on `torch-geometric`. For CUDA-specific or OS-specific wheels, follow the official guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html. The CLI will report a clear error if `torch_geometric` is missing at run time.

## Development

```bash
pip install -e ".[dev,all]"
make setup-dev      # installs pre-commit hooks
make test           # pytest
make lint           # ruff + black + isort checks
make format         # apply black + isort
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow.

## License & citation

MIT. See [LICENSE](LICENSE) (added on request).

Originally built by Arvind Tawker as an IRIS-HEP fellowship project. If you use it in research, you can cite the repo directly:

```bibtex
@software{hep_ml_templates,
  title  = {hep-ml-templates: A Modular Machine Learning Framework for High Energy Physics},
  author = {Tawker, Arvind},
  year   = {2025},
  url    = {https://github.com/livaage/hep-ml-templates}
}
```
