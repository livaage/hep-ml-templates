Here’s the complete, unrendered Markdown for a comprehensive README you can paste into VS Code.

````markdown
# HEP-ML-Templates

A modular, plug-and-play machine learning framework for **High Energy Physics (HEP)**.

**HEP-ML-Templates** provides a library of *blocks*—swappable components for data loading, preprocessing, modeling, evaluation, and full pipelines—so you can switch datasets or models with minimal code changes and zero vendor lock-in. It is designed to end one-off, ad-hoc pipelines and make results easier to reproduce and compare.

> **Status:** Production-ready with comprehensive validation, beginner-tested setup, and real-world integration case studies.

---

## Table of Contents

- [Why this library](#why-this-library)
- [Core ideas & architecture](#core-ideas--architecture)
- [Quick start (30 seconds)](#quick-start-30-seconds)
- [Installation & extras](#installation--extras)
- [Three common workflows](#three-common-workflows)
- [The `mlpipe` CLI](#the-mlpipe-cli)
- [Optional: `mlpipe-manager` CLI](#optional-mlpipe-manager-cli)
- [Configuration & overrides](#configuration--overrides)
- [Data splitting (train/val/test & time series)](#data-splitting-trainvaltest--time-series)
- [Available components (blocks)](#available-components-blocks)
- [Tutorials](#tutorials)
  - [Tutorial 1 — Scaffold a standalone project](#tutorial-1--scaffold-a-standalone-project)
  - [Tutorial 2 — Integrate a block into existing code (3-line pattern)](#tutorial-2--integrate-a-block-into-existing-code-3line-pattern)
- [Project layout](#project-layout)
- [Validation & beginner experience](#validation--beginner-experience)
- [Design principles](#design-principles)
- [Contributing](#contributing)
- [FAQ & troubleshooting](#faq--troubleshooting)
- [License & acknowledgments](#license--acknowledgments)

---

## Why this library

- **True modularity.** Swap any compatible block (dataset, model, preprocessing step, evaluation) without re-writing your pipeline.
- **Minimal changes.** Typical integration is ~3 lines to add/swap a dataset or model.
- **Beginner-ready.** New users have successfully set up all core models in *under 10 seconds per model* on average in usability tests.
- **Production quality.** Comprehensive validation across models, datasets, and installation “extras,” with clear docs and examples.

---

## Core ideas & architecture

HEP-ML-Templates revolves around four concepts:

1. **Blocks** — Self-contained Python classes with a consistent API (e.g., `ModelBlock`, `DataLoaderBlock`, preprocessing, evaluation). Blocks declare `build`, `fit`, `predict`/`transform`, etc., and hide library-specific details behind a unified interface.

2. **Registry** — A discovery mechanism that lets code and configs refer to blocks by name. Blocks register themselves with:
   ```python
   from mlpipe.core.registry import register
   @register("model.decision_tree")
   class DecisionTreeModel(ModelBlock):
       ...
````

In configs, you instantiate the same block via a short name:

```yaml
block: model.decision_tree
max_depth: 10
random_state: 42
```

3. **Configuration** — YAML (and CLI overrides) drive reproducible experiments. You can keep code stable and iterate entirely through configs.

4. **Extras system** — Curated “extras” map to concrete file sets (blocks/configs/data) you can selectively install into any target directory. Discovery, validation, preview, and installation are available both **embedded** in `mlpipe` and via an optional **`mlpipe-manager`** helper.

---

## Quick start (30 seconds)

```bash
# 1) Clone & install the core library (editable for development)
git clone https://github.com/livaage/hep-ml-templates.git
cd hep-ml-templates
pip install -e .

# 2) See what’s available
mlpipe list-extras

# 3) Scaffold a project with an XGBoost model + evaluation
mlpipe install-local model-xgb evaluation --target-dir ./my-hep-project
cd ./my-hep-project && pip install -e .

# 4) Run the default pipeline (override components on the fly)
mlpipe run --overrides model=xgb_classifier
# (Outputs training progress and evaluation metrics; values depend on your data/setup)
```

> Prefer a “manager style” UX? See the optional [`mlpipe-manager`](#optional-mlpipe-manager-cli).

---

## Installation & extras

Two common modes:

* **Develop the library itself** (inside the repo):

  ```bash
  cd hep-ml-templates
  pip install -e .
  ```

* **Use the library from another project** (install blocks into your codebase):

  ```bash
  # From your project's directory
  mlpipe install-local model-random-forest data-higgs-100k evaluation --target-dir .
  pip install -e .
  ```

### Embedded extras manager (via `mlpipe`)

Key commands:

```bash
mlpipe list-extras                     # Discover everything available
mlpipe extra-details model-xgb         # See exactly what's in an extra
mlpipe preview-install model-xgb preprocessing
mlpipe install-local model-xgb ./proj --target-dir
mlpipe validate-extras                 # Sanity-check mappings/files
```

### Extras categories (as of Aug 29, 2025)

* **Complete Pipelines** — end-to-end workflows
* **Individual Models** — single algorithms (traditional ML & neural nets)
* **Algorithm Combos** — (model + preprocessing) bundles
* **Component Categories** — preprocessing, evaluation, feature engineering
* **Data Sources** — (e.g., HIGGS, demo CSV) loaders/configs
* **Special** — comprehensive “all” bundle

> The extras system has been validated and expanded; discovery, preview, and installation are consistent and complete across categories.

---

## Three common workflows

1. **Rapid prototyping**
   Experiment with models and datasets in one place using config/CLI overrides.

2. **Standalone project scaffolding**
   Install selected extras into a clean directory so your project is self-contained and shareable (no external dependency on the templates repo structure).

3. **Integrate into existing code**
   Drop in a single block (e.g., a data loader or model) and change only a couple of lines—without disturbing the rest of your script/notebook.

---

## The `mlpipe` CLI

High-level actions:

```bash
# Discovery
mlpipe list-extras
mlpipe extra-details <extra-name>

# Safety
mlpipe validate-extras

# Planning
mlpipe preview-install <extras...>

# Scaffolding
mlpipe install-local <extras...> --target-dir ./my-project

# Execution
mlpipe run --overrides model=xgb_classifier data=higgs_100k
```

Common override patterns shown below in [Configuration & overrides](#configuration--overrides).

---

## Optional: `mlpipe-manager` CLI

Prefer a separate, dedicated tool for extras?

```bash
mlpipe-manager list
mlpipe-manager validate
mlpipe-manager details model-xgb
mlpipe-manager preview model-xgb preprocessing
mlpipe-manager install model-xgb ./my-project
```

Functionality mirrors the embedded `mlpipe` extras commands; choose whichever you prefer.

---

## Configuration & overrides

Pipelines are config-first and override-friendly.

### Block discovery (examples)

```bash
mlpipe list-extras
mlpipe extra-details model-random-forest
```

### Run with component overrides

```bash
# Swap model
mlpipe run --overrides model=decision_tree

# Swap dataset and feature engineering together
mlpipe run --overrides data=csv_demo feature_eng=demo_features

# Combine multiple overrides (example: pick model and a split strategy)
mlpipe run --overrides model=xgb_classifier preprocessing=train_val_test_split
```

### Parameter overrides (dotted keys)

```bash
# Override a parameter on the model block at runtime
mlpipe run --overrides model=xgb_classifier model.params.max_depth=8
```

### Example config snippet

```yaml
# configs/model/decision_tree.yaml
block: model.decision_tree
max_depth: 10
criterion: gini
random_state: 42
```

---

## Data splitting (train/val/test & time series)

Built-in splitting utilities support:

* Random & stratified splits
* Train/val/test in one shot
* Time-series-aware splits (no shuffling, order preserved)
* Three usage styles: convenience function, class-based, or full pipeline integration

**Convenience function:**

```python
from mlpipe.blocks.preprocessing.data_split import split_data
splits = split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, stratify=True)
X_train, y_train = splits['train']
X_val,   y_val   = splits['val']
X_test,  y_test  = splits['test']
```

**Class-based:**

```python
from mlpipe.blocks.preprocessing.data_split import DataSplitter

config = {
  'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
  'stratify': True, 'random_state': 42,
}
splitter = DataSplitter(config)
splits = splitter.fit_transform(X, y)
```

**Pipeline integration via overrides:**

```bash
mlpipe run --overrides preprocessing=train_val_test_split
```

**Pre-configured YAMLs:**

```yaml
# 80/20
train_size: 0.8
test_size: 0.2
shuffle: true
random_state: 42
```

```yaml
# 70/15/15 stratified
train_size: 0.7
val_size: 0.15
test_size: 0.15
stratify: true
shuffle: true
random_state: 42
```

```yaml
# time series (no shuffle)
train_size: 0.7
val_size: 0.15
test_size: 0.15
time_series: true
shuffle: false
```

**Install via extras:**

```bash
mlpipe install-local data-split-validation ./my-project
# or
mlpipe install-local data-split-timeseries ./my-project
```

---

## Available components (blocks)

> Names below are representative; use `mlpipe list-extras` and `mlpipe extra-details <name>` to see the exact identifiers and what each extra installs.

**Data sources**

* HIGGS (HEP benchmark)
* Demo tabular CSV
* Universal CSV loader

**Preprocessing & splitting**

* Standard scaler
* Feature engineering demos
* Train/test, train/val/test, and time series splitting utilities

**Traditional ML models**

* Decision Tree
* Random Forest
* SVM
* XGBoost
* MLP (tabular NN)
* Voting ensemble

**Neural & graph models (extended)**

* Autoencoders (vanilla, variational) — *torch*
* CNN — *torch*
* Transformers for HEP sequences — *torch*
* GNNs (GCN, GAT, etc.) — *PyG/torch*

**Complete pipelines**

* Pre-assembled end-to-end workflows (model + preprocessing + evaluation)

> Extras are organized across categories (complete pipelines, individual models, algorithm combos, component categories, data sources, special “all” bundle) and have been validated with comprehensive coverage.

---

## Tutorials

### Tutorial 1 — Scaffold a standalone project

1. **Discover components**

```bash
mlpipe list-extras
```

2. **Create a project with Random Forest + HIGGS + evaluation**

```bash
mlpipe install-local model-random-forest data-higgs-100k evaluation --target-dir ./my-first-hep-analysis
cd ./my-first-hep-analysis && pip install -e .
```

3. **Run the default pipeline**

```bash
mlpipe run
```

4. **Experiment by swapping components**

```bash
# Add new components
mlpipe install-local model-xgb data-demo-tabular .

# Switch both dataset and model in one command
mlpipe run --overrides model=xgb_classifier data=demo_tabular
```

---

### Tutorial 2 — Integrate a block into existing code (3-line pattern)

**Before** (traditional scikit-learn):

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
pred = model.predict_proba(X_test_scaled)[:, 1]
```

**After** (drop-in block with \~3 lines changed):

```python
from mlpipe.blocks.model.ensemble_models import RandomForestBlock
from mlpipe.blocks.preprocessing.standard_scaler import StandardScaler

config = {'n_estimators': 100, 'random_state': 42}
model = RandomForestBlock()
model.build(config)
model.fit(X_train, y_train)                  # Preprocessing handled internally
pred = model.predict_proba(X_test)[:, 1]
```

**Swap models by changing imports + build config:**

```python
# XGBoost
from mlpipe.blocks.model.xgb_classifier import XGBClassifierBlock
model = XGBClassifierBlock()
model.build({'n_estimators': 200, 'learning_rate': 0.1})
```

---

## Project layout

```
hep-ml-templates/
├── configs/                        # Default YAML configurations
│   ├── data/                       # Data loaders (e.g., HIGGS, CSV demo)
│   ├── model/                      # Model defaults (e.g., xgb, rf, dt, mlp)
│   └── preprocessing/              # Splitting, scaling, feature eng
├── src/
│   └── mlpipe/
│       ├── blocks/                 # Modular, swappable components
│       │   ├── data/
│       │   ├── model/
│       │   └── preprocessing/
│       ├── cli/                    # CLI entry points & local install
│       │   ├── main.py             # `mlpipe` commands
│       │   ├── manager.py          # `mlpipe-manager` (optional helper)
│       │   └── local_install.py    # Extras mapping + validation helpers
│       └── core/                   # Interfaces, registry, pipeline logic
├── comprehensive_documentation/    # Master index, validation, case studies, guides
├── pyproject.toml                  # Project metadata & CLI entry points
└── README.md                       # You are here
```

---

## Validation & beginner experience

* **Comprehensive testing** across the six core models with unified evaluation.
* **Beginner usability**: new users successfully set up every core model; average time per model is *under 10 seconds* in testing; overall experience rated “extremely easy.”
* **Real-world integration**: HIGGS benchmark case study demonstrates minimal code changes (≈3 lines) to switch datasets in an existing workflow.
* **Documentation hub**: a “master index” collects all reports, guides, and case studies in one place.

> To keep documentation flexible and future-proof, this README **does not** prescribe specific metric values (AUC/accuracy) that users “should” see. Commands produce training/evaluation output appropriate to your environment and data.

---

## Design principles

* **Single, unified interface** for blocks (models, data loaders, preprocessing, evaluation).
* **Config-first** workflows; code remains stable while you iterate through YAML.
* **Separation of concerns**: data handling, model definition, evaluation, and orchestration are decoupled.
* **Selective installation** via extras; preview, validate, and install only what you need.
* **Reproducibility** with explicit seeds, consistent splits (including stratified & time series), and clear configs.
* **Reversibility**: you can swap components in and out cleanly without vendor lock-in.

---

## Contributing

We welcome contributions—new models, datasets, preprocessing utilities, evaluation blocks, docs, and examples.

**Add a new model (outline):**

1. **Implement the block**
   Create a class that follows the model interface and register it:

   ```python
   from mlpipe.core.interfaces import ModelBlock
   from mlpipe.core.registry import register

   @register("model.my_cool_model")
   class MyCoolModel(ModelBlock):
       # implement build, fit, predict/predict_proba
   ```
2. **Provide a default config**
   Add `configs/model/my_cool_model.yaml` with sensible defaults:

   ```yaml
   block: model.my_cool_model
   some_param: 25
   ```
3. **Expose via extras**
   Update the extras mapping to include your files/configs (use helper functions for clean definitions).
4. **Validation**
   Run the extras validation tooling and library tests; ensure discovery/preview/install all work.
5. **Documentation**
   Add short usage notes and, if relevant, a tutorial or example.
6. **Open a PR**
   Follow coding style, include tests, and link to any new docs/examples.

See `CONTRIBUTING.md` for full guidelines, dev environment, testing, and review process.

---

## FAQ & troubleshooting

**Imports fail after install**

* Ensure you’re in the project directory you installed into.
* Run `pip install -e .` after `mlpipe install-local`.
* Validate mappings: `mlpipe validate-extras`.

**“Model not found” or missing configs**

* List what exists: `mlpipe list-extras`.
* Inspect details: `mlpipe extra-details <extra-name>`.

**How do I preview what a combination would install?**

* `mlpipe preview-install model-xgb preprocessing`

**How do I switch datasets & models quickly?**

* Use `mlpipe run --overrides ...`:

  ```bash
  mlpipe run --overrides data=csv_demo feature_eng=demo_features model=decision_tree
  ```

**How do I change hyperparameters without editing YAML files?**

* Use dotted overrides:

  ```bash
  mlpipe run --overrides model=xgb_classifier model.params.max_depth=8
  ```

---

## License & acknowledgments

* License: see `LICENSE` in the repository.
* Built on the Python scientific stack (scikit-learn, XGBoost, pandas, PyTorch, PyTorch Geometric, etc.) and made possible by the HEP community.
* This work is supported by the IRIS-HEP fellowship program.

---

```

---

### Grounding notes (not part of the README you paste)

- Embedded & standalone extras management, commands (`list-extras`, `extra-details`, `validate-extras`, `preview-install`, `install-local`) and categories/validation coverage are documented in the *Library Enhancements / Extras Improvements* docs (embedded CLI and optional `mlpipe-manager`) :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}.
- Data splitting features, YAML presets, and install commands are from the enhancements docs (train/val/test, time series; convenience/class/pipeline styles) :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}.
- The 3-line integration pattern and minimal-change examples (Random Forest, XGBoost) are from the local installation & testing docs :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}.
- Registry example and config schema for models are illustrated in the model swapping results (decision tree), which shows the `@register("model.decision_tree")` usage and YAML structure; reproduced here generically without metrics per the output-removal policy :contentReference[oaicite:7]{index=7}.
- Beginner setup and validation status (under 10 seconds per model, 100% success, “extremely easy”) and production-ready claims are drawn from the master index and validation reports (worded without including specific numeric model metrics in this README, per the removal guideline) :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}.
- The “no prescriptive outputs” policy in this README follows the *Specific Output Removal* summary (keep commands, remove exact expected AUC/accuracy) :contentReference[oaicite:11]{index=11}.
- HIGGS case-study (3-line dataset swap) and real-world validation are referenced from the integration summary and master index highlights :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}.

If you want this split into “README” + a “Getting Started” guide for Read the Docs, I can also generate `docs/getting_started.md` and `docs/blocks.md` from the same sources.
```
