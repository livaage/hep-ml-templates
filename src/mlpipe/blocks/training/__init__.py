# Training block imports
try:
    from . import sklearn_trainer  # noqa: F401
except ImportError:
    pass  # scikit-learn not available

# Import PyTorch trainer if dependencies are available
try:
    from . import pytorch_trainer  # noqa: F401
except ImportError:
    # PyTorch dependencies not available, skip import
    pass
