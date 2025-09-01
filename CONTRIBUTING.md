# Contributing to HEP-ML-Templates

Thank you for your interest in contributing to HEP-ML-Templates! This document provides guidelines and instructions for contributors.

## 🚀 Quick Start for Contributors

### 1. Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/livaage/hep-ml-templates.git
cd hep-ml-templates

# Set up development environment
make setup-dev

# Verify installation
make info
```

### 2. Development Workflow

```bash
# Before making changes
make quick-check

# During development
make format        # Format your code
make lint         # Check code quality
make test-fast    # Run fast tests

# Before committing
make dev-check    # Run full checks
make pre-commit   # Run all pre-commit hooks
```

## 📋 Code Quality Standards

### Automated Linting and Formatting

We use automated tools to maintain code quality. **No manual linting scripts** - everything is automated:

- **Black**: Code formatting
- **Ruff**: Fast Python linter and formatter
- **isort**: Import sorting
- **mypy**: Type checking
- **bandit**: Security scanning
- **pydocstyle**: Docstring linting

### Pre-commit Hooks

All code quality checks run automatically via pre-commit hooks:

```bash
# Install hooks (done automatically with make setup-dev)
pre-commit install

# Run manually if needed
pre-commit run --all-files
```

### GitHub Actions CI

Our CI pipeline automatically:
- ✅ Runs all linting and formatting checks
- ✅ Executes tests across Python 3.9-3.12
- ✅ Performs security scanning
- ✅ Builds and validates packages
- ✅ Runs on Ubuntu, macOS, and Windows

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Fast unit tests
├── integration/    # Integration tests  
└── conftest.py     # Shared test fixtures
```

### Running Tests
```bash
make test           # All tests
make test-fast      # Unit tests only
make test-cov       # Tests with coverage report
```

### Test Markers
- `@pytest.mark.slow`: For slow-running tests
- `@pytest.mark.integration`: For integration tests
- `@pytest.mark.unit`: For unit tests

## 📁 Project Structure Guidelines

### Keep Main Branch Clean

- **No cleanup folders**: Use `.gitignore` to exclude temporary files
- **No manual scripts**: Use automated tools (Makefile, pre-commit, CI/CD)
- **No temporary files**: Clean development environment

### Modular Architecture

- **Core library changes**: Make improvements in `src/mlpipe/`
- **Block-based design**: Each component should be self-contained
- **Registry system**: Use `@register` decorator for new blocks
- **Interface compliance**: Follow established interfaces

## 🔧 Adding New Features

### Adding a New Block

1. **Create the block file**:
   ```python
   # src/mlpipe/blocks/{category}/{block_name}.py
   from mlpipe.core.interfaces import {Interface}
   from mlpipe.core.registry import register
   
   @register("{category}.{block_name}")
   class {BlockName}({Interface}):
       def __init__(self, config=None):
           # Implementation
       
       def {required_methods}(self):
           # Implementation
   ```

2. **Add imports** to `src/mlpipe/blocks/{category}/__init__.py`

3. **Create configuration** in `configs/{category}/{block_name}.yaml`

4. **Add tests** in `tests/unit/test_{block_name}.py`

5. **Update documentation** as needed

### Adding Dependencies

1. **Core dependencies**: Add to `dependencies` in `pyproject.toml`
2. **Optional dependencies**: Add to appropriate `optional-dependencies`
3. **Development dependencies**: Add to `dev` extra
4. **Pipeline dependencies**: Update relevant `pipeline-*` extras

## 📝 Code Style Guidelines

### Docstrings (Google Style)
```python
def function(param1: str, param2: int) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When something goes wrong.
    """
```

### Type Hints
```python
from typing import Dict, List, Optional, Any

def process_data(
    data: pd.DataFrame, 
    config: Dict[str, Any],
    target_col: Optional[str] = None
) -> List[str]:
    # Implementation
```

### Error Handling
```python
# Good: Specific exceptions with helpful messages
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found: {file_path}")

# Good: Preserve context
try:
    result = risky_operation()
except SpecificError as e:
    raise ProcessingError(f"Failed to process data: {e}") from e
```

## 🚦 Pull Request Process

1. **Create feature branch**: `git checkout -b feature/your-feature-name`

2. **Make changes** following the guidelines above

3. **Run full checks**: `make dev-check`

4. **Commit with clear message**:
   ```
   Add new XYZ feature: brief description
   
   - Detailed change 1
   - Detailed change 2
   - Fixes #issue_number
   ```

5. **Push and create PR** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots if UI changes
   - Test results if applicable

6. **Address review feedback** and ensure CI passes

## 🎯 Development Best Practices

### DO ✅
- Use automated tools (make commands, pre-commit)
- Write tests for new features
- Follow existing patterns and interfaces
- Keep commits focused and atomic
- Update documentation when needed
- Test across different Python versions locally if possible

### DON'T ❌
- Commit temporary/cleanup files
- Skip running tests before submitting
- Ignore linting/formatting errors
- Create manual linting scripts
- Break existing interfaces without discussion
- Submit large, unfocused PRs

## 🛠 Local Development Commands

```bash
# Setup
make setup-dev      # One-time setup
make info          # Check environment

# Daily development
make quick-check   # Fast checks
make format        # Format code
make test-fast     # Run unit tests

# Before committing  
make dev-check     # Full validation
make pre-commit    # Run all hooks

# CI simulation
make ci-local      # Simulate full CI locally

# Cleanup
make clean         # Remove build artifacts
```

## 📞 Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check comprehensive documentation in repository

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Git history for all contributions

Thank you for helping make HEP-ML-Templates better! 🚀
