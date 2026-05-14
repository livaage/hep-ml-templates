# Contributing

## Setup

```bash
git clone https://github.com/livaage/hep-ml-templates.git
cd hep-ml-templates
make setup-dev    # installs dev + all extras, registers pre-commit hooks
```

## Workflow

```bash
make format        # black + isort
make lint          # ruff, black --check, isort --check
make test          # full test suite
make test-fast     # skip slow + integration
make dev-check     # format + lint + type-check + security + test (run before pushing)
```

Pre-commit hooks run the same checks on every commit. If a hook fails, fix the issue and re-stage — don't pass `--no-verify`.

## Adding a block

1. Implement the relevant interface from [`src/mlpipe/core/interfaces.py`](src/mlpipe/core/interfaces.py) in a new file under `src/mlpipe/blocks/<category>/`.
2. Decorate the class with `@register("category.name")` so the runtime can find it.
3. Add a default YAML config under `configs/<category>/<name>.yaml`.
4. Import the new module from `src/mlpipe/blocks/<category>/__init__.py` so the registry sees it.
5. If the block introduces new dependencies, add them to the appropriate `[project.optional-dependencies]` entry in `pyproject.toml` (and to any `pipeline-*` bundle it belongs in).
6. Add tests under `tests/`.

## Code style

- Black + isort + ruff (configured in `pyproject.toml`).
- Type hints on public functions.
- Google-style docstrings on public APIs; `pydocstyle` is enforced via ruff.
- Avoid manual lint scripts — extend the Makefile or pre-commit config instead.

## Pull requests

- Branch off `main`, keep PRs focused.
- Run `make dev-check` locally.
- Reference any related issue in the description.
- CI runs lint, type-check, security scan, and tests on Linux/macOS/Windows across supported Python versions.
